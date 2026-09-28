# Map attention during Qwen3-VL inference

真实自回归生成过程中，截取指定 decoder layer/head 对 image tokens 的 attention，映射回地图 patch 网格。不会重输生成后的 reasoning，不使用文本推测位置，不做遮挡实验。

## 运行

在 `C:\Users\junyhuang\Thesis` 下运行（已有 VisionGRPO 环境）：

```powershell
& C:/Users/junyhuang/.conda/envs/VisionGRPO/python.exe VLM_adaptation/Evaluation_scripts/Attention_visualization/attention_runner.py --output VLM_adaptation/Evaluation_scripts/Attention_visualization/outputs/test400 --layers 20 31 --heads 0 1 2 3
```

默认基座 `Qwen/Qwen3-VL-8B-Thinking`，adapter 为 `MapWise_scaling1000_GRPO_Qwen3VL8B_BS2GA8_epoch12/checkpoints/checkpoint-2500`。这与 checkpoint 的 `base_model_name_or_path` 一致。先运行无 adapter 的 baseline，再在同一基座上加载 adapter。

默认采用 NF4 4-bit、BF16 计算以适应本机 20 GB GPU；两个模型采用相同量化。`--bf16` 可关闭量化，但需要更多显存。默认只用本地缓存，下载需显式传 `--allow-download`。脚本不依赖 Unsloth 的推理补丁。所有 decoder 层固定使用原生 eager，使捕获 layer/head 的选择不会改变推理内核；vision encoder 使用 SDPA。该配置可能与原有加速推理产生浮点数值差异，输出元数据记录配置。

参数：`--questions batch.json`、`--adapter PATH`、`--layers 8 16 24 35`、`--heads 0 1 ... 31`、`--max-new-tokens 3072`、`--max-pixels 1638400`。层/head 从 0 开始；负 layer 表示从末尾计数。所有 36 个 decoder layers 都在线汇总完整 inference 的 32 个 query-head 均值；`--layers` 仅决定哪些层额外保存逐步骤、逐 head 的 float32 诊断数据，默认是 8/16/24/35。这样网页可检查完整深度范围，存储量仍适合批量运行。问题批量接收、逐题执行，不进行张量 padding batch；输出目录已有相同结果时拒绝覆盖。

## 批量接口

JSON 格式见 `example_questions.json`。增加更多记录即可；image 相对 JSON 所在目录解析，也可填绝对路径。可选 `prompt` 字段完全替换默认 prompt；默认与现有 MapWise inference 的普通问题 prompt 相同。模型使用自身 chat template 的默认 Thinking 行为。ground truth 不进入 prompt。

Python API（从 Thesis 工作目录）：

```python
from VLM_adaptation.Evaluation_scripts.Attention_visualization import Question, Settings, run_comparison

pages = run_comparison(
    [Question("example", "What value range does Guangxi fall in?",
              r"C:\Users\junyhuang\Thesis\VLM_adaptation\Datasets\mapwise-dataset\china\images\with_annotations\map20_0D.png")],
    output_dir="attention_results",
    settings=Settings(layers=(20, 31), heads=(0, 1)),
)
```

测试样本来自 `Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json` 第 120 条（索引 119）。原数据不含 qa_id，按现有 inference 的命名规则得到 `mapwise_china_map20_0D_t16_idx0119`。地图来自 `Datasets/mapwise-dataset/china/images/with_annotations/map20_0D.png`，问题为 `What value range does Guangxi fall in?`。

## 输出与时间对齐

每题包含：

- `comparison.html`：离线可打开。每一帧都是 36 个 decoder layers 的直接逐元素和，不需要选择 layer。可查看单个生成步骤、所有步骤的均值、p95（反映反复出现的强关注）、max（反映任一步骤出现过的峰值）或 literal sum（反映总 attention、受生成长度影响）。逐步骤模式直接读取推理时保存的 `all_layer_steps.npz`，不是 replay reasoning。`p99 contrast` 只设定显示色标上限，不会改动 attention 数值或聚合方式。
- `full_inference_layer16.png`：默认整段 inference 叠加预览，layer 16。
- `all_layers_raw_sum.png`：对 36 个 raw full-inference layer maps 直接逐元素求和的 baseline/adapted 并排图；不做 layer weighting 或 layer normalization。为保持 baseline/adapted 的可比较性，每个 layer map 已先在本模型的全部生成步骤和 32 个 query heads 上取均值，避免生成长度差异直接决定亮度。
- `baseline/` 与 `adapted/`：各有 `response.txt`、`result.json`、`attention.npz`、`full_layer_summary.npz` 及 `all_layer_steps.npz`。后者保存所有 36 层的完整 inference 汇总，形状为 `[layer, patch_y, patch_x]`；`all_layer_steps.npz` 保存每个生成步骤的 36-layer 直接和，形状为 `[step, patch_y, patch_x]`。

NPZ 的 `attention` 为 float32 `[step, layer, query_head, patch_y, patch_x]`，层/head 的实际编号保存在 JSON。权重是每个 query head 对所有历史 keys 经过 softmax 后的 image-key 切片；它没有忽略 key/value heads。Qwen3-VL 采用 grouped-query attention：32 个 query heads 对应 8 个 key/value heads，每 4 个 query heads 共享一个 KV head，但它们各自的 query 向量和 attention 分布仍不同。由于“一个 KV head 的 attention map”不是模型 forward 中独立输出的对象，页面聚合的是实际用于加权 value vectors 的 query-head attention weights。图像位置由 `image_token_id` 提取；按 `image_grid_thw / spatial_merge_size` 恢复 Qwen 合并后的行优先网格，再映射到原图。仅支持单张静态图像，不支持视频、多图或裁剪拼接处理器。

**步骤 0**：prefill 的最后一个 prompt query，用来预测第一个输出 token。**步骤 t > 0**：输入上一个已生成 token 的 query，用来预测第 t 个输出 token。因此页面写的是“predicting token”，不是将生成 token 错当成同一步的 query。EOS 也有相应记录。捕获次数与输出 token 数必须相等，否则报错。

默认色标在两个模型、全部步骤间共享（每个 layer/head 一组范围），展示原始局部权重。另有明确标记的 local contrast 模式，只增强每帧局部可见性，不能比较绝对强弱。HTML 和 NPZ 均保留 float32 权重，避免弱 attention 在显示量化时丢失。页面两条时间轴独立，因为相同 token index 不保证相同语义阶段。热图显示 patch 格子，不伪装成精确像素级定位；视觉特征已经融合上下文，attention 也不是因果归因。

生成文本在自然遇到 EOS 前达到 token 上限时，标记 `token_limit`，不会伪称 reasoning 完整。权重在生成中捕获，HTML 在两次生成完成后渲染；当前不提供边生成边刷新的服务端界面。

## 验证

```powershell
Set-Location VLM_adaptation/Evaluation_scripts/Attention_visualization
& C:/Users/junyhuang/.conda/envs/VisionGRPO/python.exe -m unittest test_attention.py
```

测试以真实 Qwen attention 模块核对 prefill、KV cache decode、head 顺序、网格重排及 hook 清理。运行环境需要 torch、transformers（支持 Qwen3-VL）、peft、accelerate、bitsandbytes、numpy、Pillow、matplotlib。测试过的具体环境和 GPU 运行结果见 `VALIDATION.md`。
