# DSpark 在 Qwen3.6-27B（hybrid GDN）上 hidden 变 NaN / accept len 崩到 2 的定位报告

- **日期**：2026-08-03
- **分支**：`dspark-0722-nightly`
- **现象**：同一个 DSpark draft，在 DeepSpec 的离线脚本（`DeepSpec/scripts/eval_real_dspark.py`）里 7 步平均 accept len = **6.117**；换到本仓库的 vllm-ascend 上跑，accept len 只有 **~2**。draft（5 层 dense 小模型）跑完唯一一次前向后 hidden 全部变成 NaN。
- **对照组**：target 换成 Qwen3-8B 时基本正常。
- **结论**：**不是 attention 算子本身的 shape 问题，而是 DSpark proposer 计算 query block 的 `slot_mapping` 时用错了 block_size。**该错误只在 target 为 **混合架构（GDN linear attention + full attention）** 时触发 —— 正好就是 Qwen3.6-27B 与 Qwen3-8B 的唯一结构性差异。

> 说明：本报告的代码分析是完整的、可交叉验证的；但**根因判定还差一步实机确认**（见「第 4 节 快速验证」），因为我没有 NPU 环境可以跑。验证是一行 print 的事。

---

## 1. 关键前提：Qwen3.6-27B 是 hybrid GDN 模型

`DeepSpec/scripts/eval_real_dspark.py:6` 自己就写了：

> 针对 **混合注意力 target(Qwen3.6-27B / qwen3_5, 含 linear attention)** 适配

DeepSpec 的 `build_draft_config()`（`deepspec/modeling/dspark/qwen3/config.py`）是 `copy.deepcopy(target_config)`，只覆盖少数字段，所以 draft config 会原样带上 target 的 `linear_num_key_heads / linear_key_head_dim / linear_value_head_dim / mamba_ssm_dtype / full_attention_interval / attn_output_gate` 等字段。对照 HF 上同族 checkpoint（`satgeze/Qwen3.6-27B-DSpark`）的 `config.json` 可以确认这一点。

vllm-ascend 对应的判定：

- `vllm-ascend/vllm_ascend/utils.py:1413` `check_gdn_layer()` —— `hf_config.layer_types` 里出现 `"linear_attention"` 即返回 True。
- Qwen3.6-27B → **命中**；Qwen3-8B → 不命中。

hybrid 模型在 vLLM 里会产生**多个 KV cache group**（full attention 一组 + GDN/Mamba 若干组），并触发下一节的 block size 不一致。

---

## 2. 根因：KV-manager block_size ≠ Ascend kernel block_size

### 2.1 两个 block_size 的来源

Ascend 的 GQA attention backend **只支持 128 的 kernel block size**：

```
vllm-ascend/vllm_ascend/attention/attention_v1.py:138
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]
```

而 hybrid 模型为了让 attention 的 page size 和 mamba/GDN state 的 page size 对齐，vLLM 会**放大 attention 层的 KV-manager block_size**：

```
vllm/vllm/v1/core/kv_cache_utils.py:1049  unify_kv_cache_spec_page_size()
    "... first try to unify page size by increasing the block size of layers
     with smaller page size ..."
```

于是：

| target | KV-manager block_size | Ascend kernel block_size | 是否一致 |
|---|---|---|---|
| Qwen3-8B（纯 attention） | 128 | 128 | ✅ |
| Qwen3.6-27B（hybrid GDN） | > 128（256 / 512 / …） | 128 | ❌ |

### 2.2 不一致时 BlockTable 的语义会变

```
vllm/vllm/v1/worker/block_table.py:47-66
    if kernel_block_size == block_size:
        self.block_size = block_size
        self.blocks_per_kv_block = 1
        self.use_hybrid_blocks = False
    else:
        self.block_size = kernel_block_size          # ← 128
        self.blocks_per_kv_block = block_size // kernel_block_size
        self.use_hybrid_blocks = True
```

并且 `append_row()`（`block_table.py:108-113`）会把 manager block id **展开成 kernel block id**：

```python
if self.use_hybrid_blocks:
    block_ids = self.map_to_kernel_blocks(...)
```

即：**`BlockTable.get_device_tensor()` 里存的是 128-token 的 kernel block 编号**。

物理 KV cache 也是按 kernel block size reshape 的：

```
vllm-ascend/vllm_ascend/worker/model_runner_v1.py:4527-4528
    if hasattr(attn_backend, "get_supported_kernel_block_sizes") and self.use_hybrid_blocks:
        block_size = attn_backend.get_supported_kernel_block_sizes()[0]   # 128
```

attention 读的时候也是按物理 cache 的块大小：

```
vllm-ascend/vllm_ascend/attention/attention_v1.py:1243-1250   # ChunkedPrefill 分支
    num_block, block_size, _, _ = self.key_cache.shape        # block_size = 128
    key   = self.key_cache.view(num_block, block_size, -1)
    value = self.value_cache.view(num_block, block_size, -1)
```

**所以整条读路径都是 kernel block（128）语义。**

### 2.3 DSpark 写路径用的却是 manager block size

```
vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py:160
    self.kernel_block_size = int(self.draft_attn_groups[0].kv_cache_spec.block_size)
    #                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ manager block size，不是 kernel block size
```

```
vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py:236,258
    kv_block_size = int(attn_group.kv_cache_spec.block_size)      # 同样错
    copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[1,](
        block_table_ptr=gid_block_table,   # ← 里面是 kernel block id
        block_size=kv_block_size,          # ← 却按 manager block size 索引 / 取模
        ...)
```

triton kernel 里的 slot 计算：

```
vllm-ascend/vllm_ascend/ops/triton/spec_decode/utils.py:128-132
    block_num_q = query_cache_pos // block_size
    block_id_q  = tl.load(block_table_ptr + req_idx * stride + block_num_q)
    slot_q      = block_id_q * block_size + (query_cache_pos % block_size)
```

三处同时错：

1. `query_cache_pos // block_size` —— **block_table 行号取错**（应该除 128）。
2. `query_cache_pos % block_size` —— **块内偏移取错**。
3. `block_id_q * block_size` —— `block_id_q` 已经是 kernel 编号，再乘 manager 尺寸，**slot 会放大 `blocks_per_kv_block` 倍直接越界**。

### 2.4 从错误 slot 到 NaN

DSpark 的执行顺序是：

1. `precompute_and_store_context_kv()` 写 **context** 的 K/V —— 用的是 runner 直接给的 `slot_mapping`（`_per_group_slot_mappings[gid]`），**这部分是对的**；
2. 一次 non-causal 前向，query block 的 K/V 由 attention 层自己 `reshape_and_cache` 写入，用的是上面**算错的** query slot_mapping；
3. FIA 按 `seq_lens = ctx_len + block_size` 读 paged KV。

第 3 步会去读第 2 步本该写入的那些 slot，但那里从来没被写过 —— 读到的是**未初始化显存**。bf16 下随机 bit 极容易落在 NaN/Inf 上，softmax 之后整行 hidden 变 NaN，argmax 退化成常数 token，accept len 塌到 1~2。

这与"进 attention 算子时 K/V 不对 / 第一次前向后 hidden 全 NaN"的现象完全吻合。

---

## 3. 交叉验证：这个坑仓库里已经写过注释，只有 DSpark 漏了

eagle / mtp 路径**已经**针对 `has_gdn` 修过：

```
vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:1667-1673
    # NOTE: In vllm, `block_size = attn_metadata_builder.kv_cache_spec.block_size`.
    # However, in vllm-ascend, the above value can be multiple of `kernel_block_size`,
    # which is not correct for computing `slot_mapping` below.
    if self.has_gdn:
        block_size = self.kernel_block_size
    else:
        block_size = self.block_size
```

基类里 `kernel_block_size` 的**正确**取法：

```
vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:336-338
    self.kernel_block_size = (
        draft_attn_layers_dict[self.attn_layer_names[0]]
            .get_attn_backend().get_supported_kernel_block_sizes()[0]      # → 128
    )
```

**DFlash 是对的**（直接用基类的值）：

```
vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:116
    block_size=self.kernel_block_size,
```

**只有 DSpark 在 `initialize_attn_backend()` 里把它覆盖成了 manager block size。**

更讽刺的是，runner 其实已经把正确的值传进来了，DSpark 却收下后完全没用：

```
vllm-ascend/vllm_ascend/worker/model_runner_v1.py:3937-3939
    block_size = (self.kernel_block_sizes[0] if isinstance(self.kernel_block_sizes, list)
                  else self.kernel_block_sizes)
    self.drafter.initialize_attn_backend(kv_cache_config, block_size)
```

```
vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py:101
    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        #                                               ^^^^^^^^^^^^^^^^^ 从头到尾没被使用
```

> `may_reinitialize_input_batch()`（`model_runner_v1.py:3922`）在 `initialize_attn_backend`（`:3939`）之前调用，所以 `self.runner.kernel_block_sizes` / `self.runner.input_batch.block_table[gid]` 在那个时刻已经就绪，可以安全取用。

---

## 4. 快速验证（实机 1 分钟）

### 4.1 打印两个 block size

在 `vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py:160` 附近加：

```python
for g in self.draft_attn_groups:
    gid = g.kv_cache_group_id
    bt = self.runner.input_batch.block_table[gid]
    print(f"[DSPARK] gid={gid} manager_bs={g.kv_cache_spec.block_size} "
          f"kernel_bs={bt.block_size} hybrid={bt.use_hybrid_blocks}")
```

预期：

- Qwen3-8B → `manager_bs=128 kernel_bs=128 hybrid=False`
- Qwen3.6-27B → `manager_bs>128 kernel_bs=128 hybrid=True` ← **确认根因**

### 4.2 零成本旁证

用**同一个 Qwen3.6-27B** 跑 DFlash（`{"method": "dflash", ...}`）。DFlash 用的是正确的 kernel block size，如果它不出 NaN，就基本坐实了这个判定。

### 4.3 定位到具体 tensor（可选）

在 `attention_v1.py` 的 `forward_fused_infer_attention` 里，对 draft 层打印 `attn_metadata.slot_mapping` 的 min/max，与 `self.key_cache.shape[0] * self.key_cache.shape[1]`（合法 slot 上界）对比。越界即坐实。

---

## 5. 修复方案

### 5.1 主修复

`vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py`

**（a）`initialize_attn_backend()`，约 160 行**

```python
# 原：self.kernel_block_size = int(self.draft_attn_groups[0].kv_cache_spec.block_size)
self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
# block_table / slot_mapping / 物理 KV cache 都是以 *kernel block* 为单位的；
# hybrid（GDN）target 上 kv_cache_spec.block_size 会是 kernel_block_size 的整数倍。
# 参考 llm_base_proposer.py:1667-1673 里 eagle/mtp 已有的同类修复。
self._per_group_kernel_block_size = {
    g.kv_cache_group_id: int(
        self.runner.input_batch.block_table[g.kv_cache_group_id].block_size
    )
    for g in self.draft_attn_groups
}
self.kernel_block_size = self._per_group_kernel_block_size[self.kv_cache_gid]
```

**（b）`set_inputs_first_pass()`，约 236 行**

```python
# 原：kv_block_size = int(attn_group.kv_cache_spec.block_size)
kv_block_size = self._per_group_kernel_block_size[gid]
```

`BlockTable.block_size` 在两种情形下都等于"物理 cache 的块大小"（相等时 = manager，hybrid 时 = kernel），因此这两处改动**对 Qwen3-8B 完全零行为变化**，只在 hybrid target 上生效。

**（c）顺手**：把 `initialize_attn_backend()` 那个被忽略的 `kernel_block_sizes` 形参真正用起来（或删掉），避免后续再踩。

### 5.2 回归

- Qwen3-8B + `deepseek-ai/dspark_qwen3_8b_block7`：跑 `vllm-ascend/tests/e2e/pull_request/one_card/spec_decode/test_dspark.py`，baseline `[1.0, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6]` 应保持不变。
- Qwen3.6-27B + 自训 DSpark：accept len 应该从 ~2 回到接近 DeepSpec 的 6.1。
- 建议**补一个 hybrid target 的 e2e**：目前 `tests/e2e/pull_request/one_card/spec_decode/utils.py:29-34` 里 DSpark 只覆盖了 `Qwen/Qwen3-8B`，hybrid target 完全没有测试覆盖 —— 这正是本 bug 能存活至今的原因。

---

## 6. 顺带发现的其他问题（修完 NaN 之后大概率还会吃 accept len）

### 6.1 draft 的 causal / SWA 完全没接进去

- `vllm/vllm/model_executor/models/qwen3_dflash.py:210-211`
  ```python
  # NOTE: `causal` is currently unused here, but will be needed in the future
  ```
  `_resolve_layer_attention()` 解析出来的 `causal` **根本没被使用**。
- 同文件 `:90-97`：混合 sliding/full 直接抛 `NotImplementedError`，并指向 [vllm#40898](https://github.com/vllm-project/vllm/issues/40898)。说明本仓库的 vllm 快照**不包含**该 PR。
- Ascend 侧 `dspark_proposer.py:292-294` 全局设 `cad.causal = False`；而 `attention_v1.py:1315` 的 non-causal 分支**完全忽略 `self.sliding_window`**。

当前你的 DSpark draft 是 5 层全 `full_attention`（DeepSpec `build_draft_config` 强制 `layer_types = ["full_attention"] * num_draft_layers`），所以暂时不中招；但如果之后训带 SWA 的 draft，或者改跑 DFlash checkpoint（z-lab 的 Qwen3.6 DFlash 就是 4 层 sliding + 1 层 full），就会踩。

### 6.2 context K 的 RoPE 依赖「就地修改」，返回值被丢弃

```
vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_dflash.py:45-46
    tmpv = all_k_flat.clone()
    self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)   # 返回值没接
```

- 该 custom op 声明的是 `mutates_args=[]`（`vllm-ascend/vllm_ascend/ops/register_custom_ops.py:249`），语义上就不该依赖原地修改。
- triton 路径（`ops/triton/rope.py`，`tl.store` 回 `q_ptr`）确实原地写；
- 但 `vllm-ascend/vllm_ascend/ops/rotary_embedding.py:178-201` 的 `rotary_dim < head_size` 分支是 `torch.cat` 出**新 tensor** —— 那样 **context K 的 RoPE 会被静默丢弃**，accept len 会明显掉但不报错。

**待确认**：draft config 的 `partial_rotary_factor`（会被 deepcopy 从 Qwen3.6 带过来）是否为 1.0。
**建议**无论如何改成：

```python
all_k_flat, _ = self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)
```

### 6.3 `fc` 输入维度靠巧合对上

`vllm/vllm/model_executor/models/qwen3_dflash.py:336-337` 只从 `eagle_config` / `dflash_config` 里读 `target_layer_ids`：

```python
drafter_config = getattr(self.config, "eagle_config", {})
drafter_config.update(getattr(self.config, "dflash_config", {}))
```

而 DeepSpec 存的是**顶层** `target_layer_ids`，于是 `:377-382` 走了兜底分支 `num_features_to_use = config.num_hidden_layers`。你正好 5 层 draft × 5 个 target layer，**碰巧相等**（`fc` in_features = 5×5120 = 25600 ✅）。一旦改 draft 层数就会静默错位。建议加断言：

```python
assert self.fc.input_size == len(target_layer_ids) * hidden, ...
```

（附：aux layer 的索引换算是**正确**的 —— DeepSpec `extract_context_feature()` 用 `hidden_states[layer_id + 1]`，vllm-ascend `model_runner_v1.py:653-659` 也是 `tuple(i + 1 for i in ...)`，两边一致。）

### 6.4 调试代码残留

```
vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:1096-1102
    from reg_hook import register_hooks
    if int(os.environ.get("model_hook", "0")) == 1:
        register_hooks(self.model)
```

正式跑之前记得摘掉。

---

## 7. 社区相关进展

| 链接 | 内容 |
|---|---|
| [z-lab/Qwen3.6-27B-DFlash discussion #2](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash/discussions/2) | 明确指出 *"vLLM main branch doesn't support the causal SWA layers of this draft model, which will lead to low acceptance rate"*，打上 PR #40898 后接受率从 ~6-10% 提升到 ~20-70% |
| [vllm#40898](https://github.com/vllm-project/vllm/issues/40898) | 给 DFlash 加 causal SWA 支持。本仓库快照**未包含** |
| [vllm#41190](https://github.com/vllm-project/vllm/issues/41190) | Qwen3.6 hybrid GDN + spec decode + TP=2 崩溃（已由 #48245 / #49620 关闭）。同族的「hybrid target × 投机解码」问题 |
| [vllm-ascend#11163](https://github.com/vllm-project/vllm-ascend/issues/11163) | DSpark 在 vllm-ascend 的适配 RFC，可跟进 hybrid target 支持状态 |
| [satgeze/Qwen3.6-27B-DSpark](https://huggingface.co/satgeze/Qwen3.6-27B-DSpark) | 同族 DSpark checkpoint 的 config.json，可用来核对字段 |

社区目前**没有**针对「DSpark + hybrid target + Ascend kernel block size」这个具体 bug 的修复 —— 它是 vllm-ascend 独有的（kernel block size 固定 128 + virtual block splitting 是 Ascend 特有机制）。

---

## 8. 一句话总结

Ascend 的 attention kernel 只吃 128-token 的 block，所以在 hybrid（GDN）target 上，vLLM 放大后的 KV-manager block_size 会通过 `BlockTable` 的 virtual block splitting 被拆成 128 的 kernel block；整条读路径（block_table、物理 cache、FIA）都是 kernel block 语义，唯独 **DSpark proposer 自己手算 query slot_mapping 时用了 manager block_size**（`dspark_proposer.py:160` / `:236`），把 draft 的 K/V 写到了错误甚至越界的 slot，attention 于是读到未初始化显存 → hidden 全 NaN → accept len 塌到 2。eagle/mtp 和 DFlash 都已经绕过了这个坑，只有 DSpark 漏了。
