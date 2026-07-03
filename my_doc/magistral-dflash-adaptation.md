# Magistral 接入 DFlash 投机推理适配方案(main 分支 / vLLM v0.20.2 + vllm-ascend v0.20.2rc1)

> 本文基于对 `main` 分支与 `vllm-workspace-0130-magistral-eagle3` 分支(下称 0.13.0 分支)的逐文件 diff 分析:
> - main 的 `vllm/` 与上游 vLLM `v0.20.2` **完全一致**(零改动);DFlash 框架是上游原生的。
> - main 的 `vllm-ascend/` 基于上游 vllm-ascend `v0.20.2rc1`,含 DFlash proposer 等改动。
> - 0.13.0 分支的 Magistral EAGLE3 适配 = 相对上游 vLLM v0.13.0 的 **6 个文件改动**(vllm-ascend 侧除注释掉的 debug 代码外无实质改动)。

---

## 1. 结论(TL;DR)

**你的判断基本正确:DFlash 框架本身不需要重新适配。** 而且比预期更好:0.13.0 分支上为 Magistral 做的 vLLM 侧模型适配,**大部分已经被上游 v0.20.2 吸收**(Pixtral / Mistral3 的 EAGLE3 接口、投机推理对多模态目标模型的支持等)。

main 分支上真正需要做的代码改动**只有 1 处必改**(投机推理目标模型白名单),外加视 draft 模型训练方式而定的 0~2 处可选改动。

**但有一个关键前提必须先明确:0.13.0 分支上的 Magistral EAGLE3 草稿头(draft head)不能直接当 DFlash 草稿用。** EAGLE3 与 DFlash 的草稿模型架构完全不同(EAGLE3 = 自回归单层 decoder + 拼接 3 层 aux hidden states;DFlash = 并行块草稿 + 对目标模型 context hidden states 做交叉注意力式的 KV 预计算)。要让 Magistral 跑 DFlash,**必须先训练/获得一个 Magistral 版本的 DFlash 草稿模型**,这是工作量的大头,且属于训练侧而非推理适配侧。

---

## 2. 两分支适配点对照表

0.13.0 分支上 Magistral EAGLE3 的全部有效改动,与 main(v0.20.2)现状的对照:

| # | 0.13.0 分支改动 | 内容 | main 上的现状 |
|---|---|---|---|
| 1 | `vllm/config/speculative.py` | `eagle3_target_supported` 白名单加 `"transformer"`、`"mistral"` | ❌ **仍缺失**。对应 v0.20.2 的 `aux_hidden_states_supported` 列表(`vllm/config/speculative.py:905`),没有 `mistral`,且该校验同样作用于 `dflash` 方法 → **必改,见 §3.1** |
| 2 | `vllm/model_executor/models/pixtral.py` | 给 `PixtralForConditionalGeneration` 加 `SupportsEagle3` + `set_aux_hidden_state_layers` / `get_eagle3_aux_hidden_state_layers` | ✅ 上游已合入(`pixtral.py:295`、`pixtral.py:450-456`) |
| 3 | `vllm/model_executor/models/mistral3.py` | 给 `Mistral3ForConditionalGeneration` 加 aux hidden state 两个方法 | ✅ 上游已合入(`mistral3.py:368` 带 `SupportsEagle3`) |
| 4 | `vllm/entrypoints/chat_utils.py` | 注释掉 mistral tokenizer 禁止覆盖 chat_template 的报错 | ✅ v0.20.2 中该限制已不存在(`resolve_mistral_chat_template` 整个逻辑已移除/重构),无需处理 |
| 5 | `vllm/v1/worker/gpu_model_runner.py` | 注释掉 profiling 时的多模态 encoder dummy run(配合 pixtral `embed_multimodal` 返回 None 的纯文本 hack) | ⚠️ GPU runner 专属 hack。NPU 走 `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`,没有相同的 `embed_multimodal` profiling 调用,**预计不需要**;bring-up 时验证即可(见 §5) |
| 6 | `vllm/model_executor/models/llama_eagle3.py` | 仅注释掉的 vocab mapping 导出 debug 代码 | 无需迁移 |
| 7 | `vllm-ascend/.../eagle_proposer.py`(patch 被 reject) | 多模态目标模型的 embed_tokens 共享(`model.language_model.model.embed_tokens`) | ✅ 上游已内置:`vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:232-253` 通过 `supports_multimodal(model)` → `model.get_language_model()` 处理 |

DFlash 框架侧(main 已具备,无需改动):

- vLLM 核心:`vllm/config/speculative.py`(`method="dflash"`、`parallel_drafting`)、`vllm/v1/spec_decode/dflash.py`、`vllm/model_executor/models/qwen3_dflash.py`、registry 注册 `"DFlashDraftModel"`(`registry.py:576`)。
- 多模态目标模型放行:`vllm/v1/spec_decode/dflash.py:71` 的 `_raise_if_multimodal` 已 override 为 `pass`(注释注明 "Support for multimodal inputs has not been tested"),Ascend 侧 `dflash_proposer.py:262` 同样放行。
- Ascend 侧:`vllm_ascend/spec_decode/dflash_proposer.py`(`AscendDflashProposer`)、`vllm_ascend/patch/worker/patch_qwen3_dflash.py`(融合 KV 预计算的 NPU 优化)、`vllm_ascend/ops/triton/spec_decode/utils.py` 的 triton kernel、`model_runner_v1.py` 的 dflash 分发。

---

## 3. 需要的代码修改

### 3.1 【必改】vllm/config/speculative.py — 目标模型白名单加 mistral

`method in ("eagle3", "extract_hidden_states", "dflash")` 时会校验目标模型 `hf_text_config.model_type` 是否在白名单内,Magistral 的 text config `model_type == "mistral"`,不在列表中会直接抛 `ValueError`。

**文件:`vllm/config/speculative.py`(约第 905 行)**

```python
# 修改前
aux_hidden_states_supported = [
    "llama",
    "qwen",
    "minicpm",
    "gpt_oss",
    "hunyuan_vl",
    "hunyuan_v1_dense",
    "afmoe",
    "nemotron_h",
    "deepseek_v2",
    "deepseek_v3",
    "kimi_k2",
    "kimi_k25",
    "minimax_m2",
    "gemma4",
]

# 修改后:追加两项
aux_hidden_states_supported = [
    "llama",
    "qwen",
    "minicpm",
    "gpt_oss",
    "hunyuan_vl",
    "hunyuan_v1_dense",
    "afmoe",
    "nemotron_h",
    "deepseek_v2",
    "deepseek_v3",
    "kimi_k2",
    "kimi_k25",
    "minimax_m2",
    "gemma4",
    "mistral",      # Magistral / Mistral3(HF 格式权重,text_config.model_type == "mistral")
    "transformer",  # mistral 原生 consolidated 格式加载时的 model_type(0.13.0 分支验证过)
]
```

> 说明:0.13.0 分支上实际加的就是这两项。`"mistral"` 覆盖 HF 格式(`Mistral3ForConditionalGeneration`,text_config 为 mistral);`"transformer"` 覆盖 `--config-format mistral --load-format mistral` 原生格式加载的场景。用哪种格式部署就至少需要对应那一项,两个都加最稳。
>
> 注意匹配逻辑是子串 `in`(`supported_model in model_type`),所以 `"mistral"` 同时能匹配不到 `"mistral3"` 顶层 type——校验用的是 `hf_text_config`,Mistral3 的 text config type 就是 `"mistral"`,没问题。

由于本仓库直接 vendor vLLM 源码,推荐直接改源码。若希望保持 vLLM 零侵入,也可以仿照 `vllm-ascend/vllm_ascend/patch/platform/patch_speculative_config.py` 的做法 monkey-patch,但该白名单是方法内局部变量,需要整体替换所在的 `verify` 方法,不如直接改源码干净。

### 3.2 【视 draft 训练方式而定】DFlash 草稿模型的模型类

main 上 DFlash 草稿只注册了一个实现:

```
vllm/model_executor/models/registry.py:576
    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),
```

`DFlashQwen3ForCausalLM`(`vllm/model_executor/models/qwen3_dflash.py:501`)本身**与目标模型架构解耦**:hidden_size、head 数、vocab、rope_theta 等全部来自草稿自己的 config(含 `dflash_config` 覆盖,见 `qwen3_dflash.py:231`),支持 `draft_vocab_size` + `d2t` 词表映射、与目标模型共享 embed_tokens(草稿 checkpoint 不带 embedding 时自动共享,多模态目标也已处理)。

因此有两条路线:

**路线 A(推荐,零代码改动):训练 Magistral 的 DFlash 草稿时直接沿用 Qwen3 风格的 decoder block**(即带 q/k RMSNorm 的注意力,与 `DFlashQwen3Attention` 对齐)。这样:

- 草稿 `config.json` 写 `"architectures": ["DFlashDraftModel"]`,`hidden_size`/`num_attention_heads`/`num_key_value_heads`/`head_dim`/`rope_theta`/`vocab_size` 按 Magistral 目标模型和训练配置填;
- vLLM / vllm-ascend **一行代码都不用改**(除 §3.1),`patch_qwen3_dflash.py` 的 NPU 融合 KV 优化也直接生效。

**路线 B(若草稿必须是 Mistral 风格,即注意力无 q/k norm):新增一个草稿模型类。**

1. 新增 `vllm/model_executor/models/mistral_dflash.py`:拷贝 `qwen3_dflash.py`,把 `DFlashQwen3Attention` 中的 `q_norm`/`k_norm` 去掉(或换成 `nn.Identity()`),类名改为 `DFlashMistralForCausalLM` / `DFlashMistralModel`。
2. 注册,`vllm/model_executor/models/registry.py`(_SPECULATIVE_DECODING_MODELS 区域,576 行附近):

   ```python
   "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),
   "DFlashMistralDraftModel": ("mistral_dflash", "DFlashMistralForCausalLM"),   # 新增
   ```

   草稿 `config.json` 的 `architectures` 写 `["DFlashMistralDraftModel"]`。
3. 同步适配 NPU 优化 patch:`vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_dflash.py` 的 `precompute_and_store_context_kv` 里有 `k_norm_layer = self.layers[i].self_attn.k_norm`(第 34 行附近)——新建 `patch_mistral_dflash.py` 对 `DFlashMistralModel` 打同样的 patch 但跳过 k_norm 步骤,并在 `vllm-ascend/vllm_ascend/patch/worker/__init__.py` 中 import 注册。

> 结论:优先选路线 A,把差异吸收在训练侧,推理侧零新增代码。

### 3.3 【条件改动】若草稿要用 aux hidden states 模式

上游 DFlash 草稿的 `dflash_config.use_aux_hidden_state` 默认为 `True`(`vllm/v1/spec_decode/dflash.py:282`),但 **Ascend runner 目前对 dflash 从不开启 aux hidden states 输出**:

```
vllm-ascend/vllm_ascend/worker/model_runner_v1.py:541
    if self.speculative_config.method == "eagle3":
        self.use_aux_hidden_state_outputs = self.drafter.eagle3_use_aux_hidden_state
    elif self.speculative_config.method == "extract_hidden_states":
        self.use_aux_hidden_state_outputs = True
    # ← 没有 dflash 分支,dflash 始终 False(目标模型只喂最后一层 hidden states)
```

也就是说 main 上能跑通的 Qwen3 DFlash 走的是"最后一层 hidden states"模式(草稿 config 中 `dflash_config: {"use_aux_hidden_state": false}`)。**Magistral 的 DFlash 草稿按同样方式训练/导出即可,无需改代码。**

如果训练侧坚持要 EAGLE3 式的 3 层 aux hidden states,则需在上面位置加一个分支:

```python
elif self.speculative_config.method == "dflash":
    assert isinstance(self.drafter, AscendDflashProposer)
    self.use_aux_hidden_state_outputs = self.drafter.eagle3_use_aux_hidden_state
```

并回归验证 `dflash_proposer.py` 中 `_dflash_hidden_states` buffer 的宽度(当前按 `self.hidden_size` 分配)与 `combine_hidden_states` 的 fc 融合路径。目标模型侧无需改:v0.20.2 的 `Mistral3ForConditionalGeneration` / `PixtralForConditionalGeneration` 已实现 `SupportsEagle3` 接口。**建议不走这条路,与现有 Qwen3 DFlash 保持一致。**

---

## 4. 必须准备的模型资产(非代码,但是关键路径)

1. **Magistral DFlash 草稿权重**:0.13.0 分支的 EAGLE3 头(自回归、依赖 `d2t`/`t2d` vocab mapping、拼 3 层 aux hidden)与 DFlash 草稿(并行块解码、`precompute_and_store_context_kv` 交叉 KV、mask token 机制)结构不同,**不可复用**,需按 DFlash 训练流程(参考 z-lab 的 `Qwen3-8B-DFlash-b16`)为 Magistral 重训。
2. 草稿 `config.json` 要点(对照 z-lab Qwen3 DFlash checkpoint):
   - `"architectures": ["DFlashDraftModel"]`(路线 A)
   - `hidden_size` 必须等于 Magistral 目标模型 hidden_size(交叉 KV 直接吃目标 hidden states)
   - `rope_theta` 等位置编码参数与训练时一致
   - `dflash_config`: `{"use_aux_hidden_state": false, ...}`(与 §3.3 对齐;block 大小等训练超参也放这里)
   - 若用裁剪词表:`draft_vocab_size` + 权重中带 `draft_id_to_target_id`(d2t),`DFlashQwen3ForCausalLM.compute_logits` 已支持映射回目标词表
   - 草稿不含 embed_tokens 时自动共享目标模型(多模态目标取 `language_model`,已支持)
3. `speculative_config` 中 `"method": "dflash"` 显式指定,或让草稿模型路径名包含 `dflash` 触发自动识别(`vllm/config/speculative.py:591`)。

启动示例(对照 `vllm-ascend/tests/e2e/singlecard/spec_decode/test_v1_spec_decode.py` 的 dflash 用例):

```bash
vllm serve /path/to/Magistral-Small \
    --tokenizer-mode mistral \
    --speculative-config '{
        "method": "dflash",
        "model": "/path/to/Magistral-DFlash-draft",
        "num_speculative_tokens": 8
    }'
```

---

## 5. Bring-up 验证清单(按顺序)

1. **不带投机推理,先在 main 上把 Magistral 目标模型本身跑通**(v0.20.2 对 Mistral3/Pixtral 的支持比 0.13.0 完整,0.13.0 上的纯文本 hack——`pixtral.embed_multimodal return None`、`gpu_model_runner` 注释 mm profiling——是 GPU runner 场景的 workaround;NPU 的 `model_runner_v1.py` 没有相同代码路径,预期不需要。若 profiling 阶段视觉塔报错,再评估是否在 Ascend runner 侧做等价跳过)。
2. 加上 §3.1 白名单修改,用 **EAGLE3**(0.13.0 训练的 Magistral eagle3 头)在 main 上先验证投机推理链路:embed 共享、多模态 target 的 `get_language_model()` 路径、rejection sampler。这一步能独立于 DFlash 草稿训练提前排雷,因为它复用的是与 dflash 相同的 `llm_base_proposer` 基础设施。
3. Magistral DFlash 草稿训练完成后,替换 `method: "dflash"` 跑通;对照 `test_v1_spec_decode.py::test_dflash_acceptance` 的写法补一个 Magistral 的 acceptance 用例,关注逐位置接受率曲线是否与 Qwen3 DFlash 基线形态一致。
4. 多模态输入(图文混合)+ DFlash 上游标注 "not tested":若业务只需纯文本,建议在部署层限制 `--limit-mm-per-prompt '{"image": 0}'`;若需要图文,需专项验证 mrope/占位符与并行草稿 slot 的交互(`llm_base_proposer._raise_if_mrope` 对 mistral3 不触发,因为 pixtral 不用 mrope,但建议实测)。

---

## 6. 工作量小结

| 事项 | 类型 | 工作量 |
|---|---|---|
| `speculative.py` 白名单 + `"mistral"`, `"transformer"` | 必改代码 | 2 行 |
| Magistral DFlash 草稿模型训练 | 模型训练 | **大头,决定整体进度** |
| 草稿模型类(路线 A) | 代码 | 0 行 |
| 草稿模型类(路线 B,Mistral 风格 block) | 代码 | 新文件 ~600 行(拷贝改)+ registry 1 行 + ascend patch 1 个文件 |
| aux hidden states 模式(不推荐) | 代码 | runner 3 行 + 回归 |
| 0.13.0 其余 5 个文件的改动 | — | 已被上游 v0.20.2 吸收或不再需要,无需迁移 |
