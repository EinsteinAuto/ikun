# dlopen SO开发计划 — 从日志到代码

> 基于 comp168 docker (2d5232c5) 日志分析 + 真机代码 tree (不带 --depth)
> 原则：upstream已有的搬过来，接口对上，不允许fallback，不允许全新开发

---

## 一、真机调用链现状（qwen3_5.py imports）

qwen3_5.py 声明了 **11个** corex SO模块的 import：

| # | 模块名 | prebuilt .so | .cu源码 | build脚本 | qwen3_5.py调用点 | 状态 |
|---|--------|-------------|---------|-----------|-----------------|------|
| 1 | corex_gdn_causal_conv | ✅ | ✅ | ✅ | L1158: conv更新 | **就绪** |
| 2 | corex_gdn_gated_norm | ✅ | ✅ | ✅ | L848: 反向norm | **就绪** |
| 3 | corex_gdn_beta_decay | ✅ | ✅ | ✅ | L1215: 衰减计算 | **就绪** |
| 4 | corex_gdn_qk_map | ✅ | ✅ | ✅ | L1258: QK映射 | **就绪** |
| 5 | corex_gdn_packed_decode | ✅ | ✅ | ✅ | L1195: 打包解码 | **就绪** |
| 6 | corex_attn_head_rms_norm | ✅ | ✅ | ✅ | L1322: 头归一化 | **就绪** |
| 7 | corex_moe_exact_reduce | ✅ | ✅ | ✅ | L1707: MoE精确归约 | **就绪** |
| 8 | corex_moe_weight_gather | ✅ | ✅ | ✅ | L1681: 权重收集 | **就绪** |
| 9 | corex_moe_direct_routed | ✅ | ✅ | ✅ | L1659: 直接路由MoE | **就绪** |
| 10 | corex_moe_topk_softmax | ✅ | ✅ | ✅ | L1621: topk+softmax | **就绪** |
| 11 | corex_moe_index_combine | ❌ 无prebuilt | ✅ | ✅ | L1719: 索引合并 | **需在docker build编译** |

## 二、prebuilt有但qwen3_5.py没引用的SO

| 模块名 | prebuilt | .cu源码 | qwen3_5.py引用 | 说明 |
|--------|---------|---------|---------------|------|
| corex_block_major_kv_transfer | ✅ | ✅ | ❌ | block_major_kv_cache.py用 |
| corex_fused_paged_prefill | ✅ | ✅ (split4版) | ❌ | paged_attn.py用 |
| corex_paged_kv_gather | ✅ | ✅ | ❌ | paged_attn.py用 |

## 三、有.cu但无prebuilt的模块

| 模块名 | .cu源码 | 说明 | 行动 |
|--------|---------|------|------|
| corex_gdn_chunk_recurrent | ✅ (10807字节) | GDN prefill chunked recurrent | **需precompile，可能是NaN修复的关键** |
| corex_fused_paged_prefill_split4 | ✅ (20172字节) | 分4路prefill attention | prebuilt有 corex_fused_paged_prefill (名字不同) |
| corex_moe_index_combine | ✅ (5554字节) | patch_ops.sh已有编译步骤 | **Docker内编译** |
| corex_query_tiled_paged_prefill | ✅ (20409字节) | Q-tiled prefill | 当前paged_attn.py的Python版替代 |

## 四、comp168日志揭示的关键差距

comp168（竞争对手sub168）的Docker工作正常：
- GDN：用 corex_gdn.so 的fused kernel，**无NaN**
- MoE：用自己的 topk_softmax 实现 + WMMA group_gemm，**不依赖 ixf_F.vllm_moe_topk_softmax**
- 权重：17.35 GB（我们16.23 GB）
- model_runner.py: 用base镜像原版(1074行)，不是我们的1119行版

我们的Docker(sub655)的问题：
- GDN：99.98% NaN → nan_to_num → 输出垃圾
- MoE：fallback到PyTorch loop → 约50x慢
- 服务器最终崩溃 → Connection refused → 881个replay请求全失败

## 五、现在的代码量够不够？

```
qwen3_6_scripts/
├── 15个 corex_*.cu 文件       (总计 ~115K 字节 CUDA源码)
├── 14个 build_corex_*.sh      (编译脚本)
├── 13个 prebuilt/*.so          (已编译二进制)
├── qwen3_5.py                  (1700+行，模型实现)
├── patch_ops.sh                (部署脚本)
├── paged_attn.py               (paged attention)
├── serving_chat.py + protocol.py + api_server.py (serving层)
├── vendor_overrides/           (vllm核心override，6文件)
└── ...

ex_engine/
├── csrc/                       (C++ bridge代码，24个文件)
├── python/                     (Python bridge代码，7个文件)
├── xllm_kernels/               (xllm上游kernel，8个文件)
└── xllm_layers/ + xllm_models/ (xllm上游层/模型实现)

upstream_ref/
├── ds_vllm/                    (最新vllm参考实现)
├── xllm/                       (xllm完整参考)
├── fla/                        (flash-linear-attention参考)
└── vllm_gdn/                   (vllm GDN参考实现)
```

**回答你的问题：代码数量是够的。** 15个.cu、13个prebuilt .so、qwen3_5.py已经完整引用了所有11个import。问题不是代码数量，是：

1. **corex_moe_index_combine.so 没有prebuilt** — 需要在docker build时在线编译
2. **corex_gdn_chunk_recurrent.so 没有prebuilt** — 10K字节的GDN prefill kernel，可能是解决NaN的关键
3. **patch_ops.sh 只编译了 moe_index_combine** — 其余12个走prebuilt安装

## 六、下一步行动（代码开发，不是推理）

### 立即要做的3件事：

**1. 把 corex_gdn_chunk_recurrent 加入 prebuilt 或 patch_ops.sh 编译链**

这个.cu存在（10807字节），build脚本也存在，但既没有prebuilt .so，也没在patch_ops.sh里编译。真机上需要：

```bash
# 在你的BI-V100真机上：
cd /home/dylan/project_6/qwen3_6_scripts
bash build_corex_gdn_chunk_recurrent.sh /usr/local/corex/lib/python3/dist-packages/vllm
# 如果成功，把.so拷到 prebuilt/corex-3.2.3-ivcore10/
```

**2. qwen3_5.py GDN prefill路径需要对接 chunk_recurrent kernel**

当前qwen3_5.py的GDN prefill fallback是纯PyTorch `_torch_chunk_gated_delta_rule`，产生NaN。corex_gdn_chunk_recurrent.cu 是 fp32 accumulation 的 kernel — 应该能解决NaN。需要在qwen3_5.py里加上对应的 import + dispatch。

**3. 把 corex_fused_paged_prefill_split4.cu precompile**

这个20K字节的kernel对应prefill attention加速，prebuilt目录有 `corex_fused_paged_prefill.so`（可能是同一个的改名），需要确认对应关系。

### 在真机上验证步骤：

```bash
# 单卡验证：
cd /home/dylan/project_6
python3 -c "
import torch
# 测试prebuilt SO能否加载
import importlib.util
spec = importlib.util.spec_from_file_location('corex_gdn_causal_conv', 
    'qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/corex_gdn_causal_conv.so')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('corex_gdn_causal_conv loaded:', dir(mod))
"
```

## 七、commit 9ff2450（能得分的版本）

这个commit不在当前仓库里。你说它是 `clean: remove build artifacts from docker context`，date Aug 12 07:58。这意味着它是在current HEAD (17fdf7e2) 之后的commit，可能在另一个branch或还没push。

**需要你执行：**
```bash
git log --all --oneline | grep 9ff2450
# 或者
git push origin main  # 如果在真机上有unpushed commits
```

## 八、ex_engine upstream搬运清单

ex_engine里有大量代码但 **没有接入 patch_ops.sh 部署链**。以下是已有但未使用的：

| 文件 | 功能 | upstream来源 | 接入状态 |
|------|------|-------------|---------|
| ex_engine/python/corex_gdn.py | GDN完整dispatch | 自己写的 | ❌ 未部署 |
| ex_engine/python/corex_moe.py | MoE完整dispatch | 自己写的 | ❌ 未部署 |
| ex_engine/python/ix_bridge.py | C++→Python bridge | 自己写的 | ❌ 未部署 |
| ex_engine/csrc/ix_full_bridge.cpp | ixformer C++桥 | 基于symbol probe | ❌ 未部署 |
| ex_engine/xllm_kernels/cuda/moe/*.cu | MoE CUDA kernels | xllm upstream | ❌ 未部署 |
| ex_engine/xllm_layers/npu_torch/*.cpp | 层实现 | xllm upstream | ❌ 未部署 |

**这些不需要重写，但接口要对上后再搬。** 特别是 ix_full_bridge.cpp 里明确说了 "MoE functions are NOT in base image"，所以 MoE 必须走 prebuilt .so + Python fallback 路线，而不是试图 dlopen 不存在的 ixformer MoE symbols。

现在的策略（13个prebuilt .so + 1个在线编译）已经是正确的路线。
