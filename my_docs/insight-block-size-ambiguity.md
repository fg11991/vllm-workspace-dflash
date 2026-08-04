# 一个 NaN 背后的抽象泄漏：DSpark 在 Qwen3.6 上接受率崩塌的定位复盘

> **面向读者**：推理团队。假定你熟悉 vLLM，不假定你做投机解码。
> **日期**：2026-08-03
> **对外产出**：[vllm-ascend#13372](https://github.com/vllm-project/vllm-ascend/pull/13372)（已提交）、[vllm-ascend#13380](https://github.com/vllm-project/vllm-ascend/issues/13380)（测试覆盖缺口）

---

## 摘要

我们自训的 DSpark draft 在 DeepSpec 离线脚本里 7 步平均 accept len **6.117**，换到 vllm-ascend 上跑同一个模型、同一道题，接受率只有 **20%**。

根因不在算子，而在一个**同名不同义的量**：`block_size` 在 vllm-ascend 里同时指代三种东西，而 DSpark 的 slot mapping 用错了其中一个。修复后同一道题接受率 **97.5%**。

生产代码最终改动是 **2 处 if/else**。但从"接受率低"走到这 2 处，中间绕过了几个值得记下来的坑。

---

## 一、现象与第一个岔路

| | accept 表现 |
|---|---|
| DeepSpec 离线脚本（同一 draft） | 7 步平均 accept len 6.117 |
| vllm-ascend，target = Qwen3-8B | 正常 |
| vllm-ascend，target = Qwen3.6-27B | 接受率 20%，draft hidden 全 NaN |

最初的怀疑是"进 attention 算子时 K/V 最后一维 shape 不对"。这个方向是错的，但它指对了**位置** —— 问题确实发生在 draft 的 attention 读 KV cache 的那一刻，只是原因不是 shape，而是**读到了从未被写入的地址**。

真正把方向掰正的是**差异法**：同一份 draft 代码，Qwen3-8B 好、Qwen3.6-27B 崩。两者的结构差异只有一条 —— **Qwen3.6 是 hybrid 模型**（Gated-DeltaNet 线性注意力层 + full attention 层交替）。于是问题变成："hybrid target 会改变 draft 的什么？"

答案是 KV cache 的分块方式。

---

## 二、根因：`block_size` 的三重含义

这是本文最想传达的一点。vllm-ascend 里至少有三个不同的量都叫 `block_size`：

| 名字 | 含义 | 谁决定 |
|---|---|---|
| **KV manager block size** | KV cache 管理器分配的逻辑块大小，`kv_cache_spec.block_size` | vLLM 按 page size 对齐需求推导 |
| **kernel block size** | 注意力算子实际能寻址的物理块大小 | 后端 `get_supported_kernel_block_sizes()` |
| **spec block size** | 投机解码一次提多少 token，`num_speculative_tokens` | 用户配置 |

前两个在**纯 attention 模型上恒等**，所以谁都不会注意到区别。但 hybrid 模型会让它们分叉：

vLLM 为了让 attention 的 page size 和 mamba/GDN state 的 page size 对齐，会**放大 attention 层的 manager block size**（`unify_kv_cache_spec_page_size`：*"unify page size by increasing the block size of layers with smaller page size"*）。而 Ascend 的 GQA 注意力后端只支持 128：

```python
# vllm_ascend/attention/attention_v1.py:138
def get_supported_kernel_block_sizes() -> list[int]:
    return [128]
```

两者一旦不等，`BlockTable` 就进入拆块模式，**把每个 manager block 展开成多个 kernel block**：

```python
# vllm/v1/worker/block_table.py:47-66
if kernel_block_size == block_size:
    self.block_size = block_size
    self.use_hybrid_blocks = False
else:
    self.block_size = kernel_block_size          # 128
    self.blocks_per_kv_block = block_size // kernel_block_size
    self.use_hybrid_blocks = True
```

此后 **block table 里存的是 kernel block 编号**，物理 KV cache 也按 128 reshape，注意力算子也按 128 读。整条链路统一在 kernel block 语义上。

**唯独 DSpark 的 slot mapping 用了 manager block size。**

```python
# 修复前：block_table 里是 kernel block 编号，却按 manager block size 索引和取模
block_num_q = query_cache_pos // block_size
block_id_q  = block_table[req_idx, block_num_q]
slot_q      = block_id_q * block_size + (query_cache_pos % block_size)
```

三处同时错：行号取错、块内偏移取错、`block_id * manager_size` 还会**越界**（block_id 已是 kernel 编号，再乘 manager 尺寸，放大 `blocks_per_kv_block` 倍）。

结果：draft 的 query block K/V 被写到错误甚至越界的 slot；attention 转头去读那些本该被写入的位置，读到的是**未初始化显存**。bf16 下随机 bit 很容易落在 NaN/Inf 上，softmax 之后整行 hidden 变 NaN。

### 一个反直觉的细节

我们的启动命令里**显式写了 `--block-size 128`** —— 正是 Ascend 支持的那个值 —— 照样触发。因为 page size 对齐发生在**用户设置之上**：你给 128，vLLM 为了对齐 mamba 还会往上放大。

> **教训**：任何自己手算 slot mapping 的代码，都不能假设用户给的 `--block-size` 就是 KV 管理器最终采用的值。

---

## 三、路标就在代码里，只是没人跟上

最"扎心"的一点：这个坑**仓库里早就有人踩过并留了注释**。

```python
# vllm_ascend/spec_decode/llm_base_proposer.py:1612-1615
# NOTE: In vllm, `block_size = attn_metadata_builder.kv_cache_spec.block_size`.
# However, in vllm-ascend, the above value can be multiple of `kernel_block_size`,
# which is not correct for computing `slot_mapping` below.
if self.has_gdn:
    block_size = self.kernel_block_size
```

eagle/mtp 路径在 [#12000](https://github.com/vllm-project/vllm-ascend/pull/12000) 里已经修过同一个问题。DFlash 也是对的（它直接用基类算好的 `self.kernel_block_size`）。

**只有 DSpark 没跟上。** 而且更微妙 —— 它不是"忘了算"，是**把基类已经算对的值覆盖成了错的**：

- `llm_base_proposer.load_model()` 已设好 `self.kernel_block_size`
- 调用顺序：`load_model()` → `initialize_attn_backend()`
- DSpark 在后者里写了 `self.kernel_block_size = int(...kv_cache_spec.block_size)`

这个发现是顺着 maintainer 的一句"Please follow the logic in `llm_base_proposer`"找到的，也是把 PR 从 +224 行压到 2 处 if/else 的关键。

> **教训**：一个 bug 在同一个仓库里出现第二次，说明第一次的修复只治了症状、没有把不变量变成结构约束。留 NOTE 注释是好的，但它只保护读到那段代码的人。

---

## 四、为什么没有任何东西报警

整条失败链路上，**没有一个 assert**：

1. slot 算错并越界 → `reshape_and_cache` 没报错
2. attention 读未初始化显存 → 没报错
3. hidden 变 NaN → 没有 NaN 检测
4. 接受率跌到 20% → 服务照常返回结果，功能上"可用"

这是投机解码特有的危险形态：**draft 完全坏掉时，服务依然正常出 token**，因为 target 会拒绝掉所有错误提案，正确性由 verify 兜住。你只会损失性能，而性能没有下界告警。

我们在 issue 里对 e2e 设计提了一条：

> 光检查"响应非空"的 smoke test 不够 —— NaN 的 draft 依然会返回 token。必须断言逐位置接受率高于基线。

> **教训**：任何"错了只掉性能不掉正确性"的优化路径，都需要一条性能下界告警。否则它坏了你不会知道。

---

## 五、第二个 bug：跨仓库签名契约

同一次排查里还修了一个独立问题，模式很典型。

`patch_mamba_utils` 把 Ascend 的 mamba postprocess kernel **换掉上游 vLLM 的同名 kernel，但调用方留在上游**：

```python
mamba_utils.postprocess_mamba_fused_kernel = postprocess_mamba_fused_kernel
```

上游后来给这个 kernel 加了 6 个参数（`state_dim_row_count_ptr`、`idx_mapping_ptr`、`CONV_STATE_DIM_FIRST` 等）。于是每次调用都 `TypeError`，**hybrid 模型服务根本起不来**。纯 attention 模型不走 mamba 路径，所以完全无感。

我们的修法不只是补齐签名，还实现了两个几行就能对齐的模式（`HAS_IDX_MAPPING`、`PRECOMPUTED_NEW_COMPUTED`），并对没实现的 DS conv layout 加了 `tl.static_assert` 拒绝，而不是静默拷错字节。

> **教训**：patch 上游函数但不 patch 调用方，等于和上游签了一份**没有编译期检查的契约**。这类地方值得有廉价的守卫（我们提了一个 AST 比对上游参数列表的单测，评审认为冗余，已按要求移除 —— 这个取舍可以再讨论）。

---

## 六、测试矩阵的盲区是按"架构族"划的

两个 bug 都逃过了 CI，原因相同：

```python
# tests/e2e/pull_request/one_card/spec_decode/utils.py
DSPARK = {"dspark": {"main": "Qwen/Qwen3-8B", "spec": "deepseek-ai/dspark_qwen3_8b_block7"}}
```

DSpark 的 e2e **只有一个纯 attention target**。而这两个 bug 都只在有 mamba/GDN 层时才出现：

- bug 1 需要 manager block size ≠ kernel block size（只有 page size 对齐才会发生）
- bug 2 需要走到 mamba postprocess（纯 attention 根本不调用）

> **教训**：测试矩阵的维度不该是"模型名"，而是"会走到哪些代码分支的架构特征"。hybrid / MLA / 纯 attention 是三条不同的路径，各挑一个代表比堆五个同族模型有用得多。

---

## 七、社区评审逮到了我们没看到的东西

值得单独记一笔。我们第一版修复是**无条件**使用 `self.kernel_block_size`。maintainer `slippersss` 指出这会搞坏 DeepSeek-V4 DSpark。查下来比他说的更严重：

```python
# vllm_ascend/attention/dsa_v1.py:230  (DSV4 用的 DSA 后端)
def get_supported_kernel_block_sizes() -> list[int]:
    return [2, 4, 8, 16, 32, 64, 128]
```

基类取的是 `[0]` —— 也就是 **2**。无条件用它会把 dsv4 的 slot mapping 彻底写坏。最终按 #12000 的模式加了 `has_gdn` 分支，非 GDN 路径与改动前逐字节一致。

> **教训**：一个只在自己场景验证过的"通用修复"，很可能是另一个场景的新 bug。上游评审的价值不在挑格式，在于它带着你没有的上下文。

---

## 八、顺带澄清一个容易搞混的点：训练 block size vs `num_speculative_tokens`

排查过程中发现这两个概念在团队里容易混。结论：

- **checkpoint 里的 `block_size` 是训练超参，vLLM 推理路径完全不读它**，也没有任何校验
- 推理时 draft 实际跑多长、Markov head 串行几步、target 每轮 verify 几个 token，**全部由 `num_speculative_tokens` 决定**

| 环节 | spec_len=7 | spec_len=12 |
|---|---|---|
| draft query block 前向长度 | 7（1 anchor + 6 mask） | 12 |
| Markov head 串行步数 | 7 | 12 |
| target 每轮 verify token 数 | 8 | 13 |

所以 draft **不会**"跑满训练块长再截断"。把 spec_len 调得比训练块长**低**是合法旋钮（块内是双向注意力，会有分布错配，但成本同时下降；z-lab 报告过他们的 Qwen3.6 DFlash 取 2~5 比取 15 端到端更快）。调得**高**才是真危险：超出训练块长的位置，其 RoPE 偏移和 mask-slot 数量模型从没见过，同样不会报错。

硬约束只在变大方向：`1 + num_speculative_tokens <= 16`（`npu_fused_infer_attention_score` TND layout 限制）。

---

## 九、如果你怀疑自己撞上了同一个坑

一行 print 就能判定：

```python
# 加在 dspark_proposer.initialize_attn_backend 里
for g in self.draft_attn_groups:
    gid = g.kv_cache_group_id
    bt = self.runner.input_batch.block_table[gid]
    print(f"[dspark] manager_bs={g.kv_cache_spec.block_size} "
          f"kernel_bs={bt.block_size} hybrid={bt.use_hybrid_blocks}")
```

`manager_bs != kernel_bs`（`hybrid=True`）就是命中。相等则是别的原因。

`releases/v0.25.1rc` 上有完全相同的代码（行号也一样），2 行即可自行 patch：

```python
# vllm_ascend/spec_decode/dspark_proposer.py:164 —— 删掉这行
- self.kernel_block_size = int(self.draft_attn_groups[0].kv_cache_spec.block_size)

# :239
- kv_block_size = int(attn_group.kv_cache_spec.block_size)
+ kv_block_size = self.kernel_block_size    # 注意：需按 has_gdn 分支，见 #13372
```

---

## 十、数据与复现条件

| | |
|---|---|
| 硬件 | Atlas 800I A2（910B）×4，TP=4，DP=1 |
| target | Qwen3.6-27B（hybrid Gated-DeltaNet） |
| draft | 内部自训 Qwen3.6-27B DSpark drafter，block=7 |
| dtype / 图模式 | bfloat16 / `--enforce-eager` |
| 其他 | `--enable-prefix-caching --block-size 128 --enable-chunked-prefill --max-num-seqs 1 --max-model-len 8192` |

单条贪心请求（`"Write a Python function to compute the nth Fibonacci number iteratively."`，`temperature=0`，`max_completion_tokens=512`）：

| 状态 | 结果 |
|---|---|
| 未修 mamba 签名 | 服务起不来 |
| 修了签名、未修 slot mapping | 接受率 **20%**，draft hidden 为 NaN，后三个位置全不接受 |
| 两个都修 | 接受率 **97.5%**，前四个位置全接受 |

参考：同一 drafter 在 DeepSpec 参考实现下 7 步平均 accept len = 6.117，与修复后一致、与修复前不一致。

> ⚠️ 上述端到端数字来自**单条 prompt、greedy**，不构成基准测试。纯 attention target（Qwen3-8B）的回归尚未在硬件上重跑 —— 代码层面该路径是恒等变换（`has_gdn=False` 时与改动前逐字节相同），但仍待 CI 确认。

---

## 附：相关链接

| 内容 | 链接 |
|---|---|
| 修复 PR | [vllm-ascend#13372](https://github.com/vllm-project/vllm-ascend/pull/13372) |
| 测试覆盖缺口 issue | [vllm-ascend#13380](https://github.com/vllm-project/vllm-ascend/issues/13380) |
| eagle/mtp 的同类先例 | [vllm-ascend#12000](https://github.com/vllm-project/vllm-ascend/pull/12000) |
| DFlash 的 causal SWA 支持（另一个坑） | [vllm#40898](https://github.com/vllm-project/vllm/issues/40898) |
| 详细 RCA（含完整代码引用） | `my_docs/dspark-qwen3.6-nan-accept-len-rca.md` |

---

## 三条带走的东西

1. **同名不同义的量是 bug 温床。** `block_size` 在这套代码里有三个含义，凡是自己手算地址的地方都要先问清"我这个是哪一个"。
2. **只掉性能不掉正确性的路径，必须有性能下界告警。** 否则它坏了没人知道 —— draft 全 NaN 的服务看起来完全正常。
3. **测试矩阵按架构特征划，不按模型名划。** 五个纯 attention 模型的覆盖度，等于一个纯 attention 模型。
