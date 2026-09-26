# Map attention during Qwen3-VL inference

真实自回归生成过程中，截取指定 decoder layer/head 对 image tokens 的 attention，映射回地图 patch 网格。不会重输生成后的 reasoning，不使用文本推测位置，不做遮挡实验。

## 运行

在 `C:\Users\junyhuang\Thesis` 下运行（已有 VisionGRPO 环境）：

```powershell
& C:/Users/junyhuang/.conda/envs/VisionGRPO/python.exe VLM_adaptation/Evaluation_scripts/Attention_visualization/attention_runner.py --output VLM_adaptation/Evaluation_scripts/Attention_visualization/outputs/test400 --layers 20 31 --heads 0 1 2 3
```

默认基座 `Qwen/Qwen3-VL-8B-Thinking`，adapter 为 `MapWise_scaling1000_GRPO_Qwen3VL8B_BS2GA8_epoch12/checkpoints/checkpoint-2500`。这与 checkpoint 的 `base_model_name_or_path` 一致。先运行无 adapter 的 baseline，再在同一基座上加载 adapter。

默认采用 NF4 4-bit、BF16 计算以适应本机 20 GB GPU；两个模型采用相同量化。`--bf16` 可关闭量化，但需要更多显存。默认只用本地缓存，下载需显式传 `--allow-download`。脚本不依赖 Unsloth 的推理补丁。所有 decoder 层固定使用原生 eager，使捕获 layer/head 的选择不会改变推理内核；vision encoder 使用 SDPA。该配置可能与原有加速推理产生浮点数值差异，输出元数据记录配置。

参数：`--questions batch.json`、`--adapter PATH`、`--layers 8 16 24 35`、`--heads 0 1 ... 31`、`--max-new-tokens 3072`、`--max-pixels 1638400`。层/head 从 0 开始；负 layer 表示从末尾计数。默认捕获 layers 8/16/24/35 的全部 32 个 query heads，页面显示同层全部 heads 的均值；逐 head float32 权重保存在 NPZ 中用于诊断，单个 head 不能代表模型的总体关注。问题批量接收、逐题执行，不进行张量 padding batch；避免 attention 内存膨胀。输出目录已有相同结果时拒绝覆盖。

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

- `comparison.html`：离线可打开，左右各自拖动/播放生成步骤，切换 layer/head，显示当前生成前缀，导出当前地图或并排 PNG。
- `first_step.png`：第一个预测步骤、首个所选 layer/head 的并排预览，不是整段推理的平均。
- `baseline/` 与 `adapted/`：各有 `response.txt`、`result.json` 和 `attention.npz`。

NPZ 的 `attention` 为 float32 `[step, layer, head, patch_y, patch_x]`，层/head 的实际编号保存在 JSON。权重仍是对全部历史 keys 归一化后的 image-key 切片，不对图像部分重新做 softmax。图像位置由 `image_token_id` 提取；按 `image_grid_thw / spatial_merge_size` 恢复 Qwen 合并后的行优先网格，再映射到原图。仅支持单张静态图像，不支持视频、多图或裁剪拼接处理器。

**步骤 0**：prefill 的最后一个 prompt query，用来预测第一个输出 token。**步骤 t > 0**：输入上一个已生成 token 的 query，用来预测第 t 个输出 token。因此页面写的是“predicting token”，不是将生成 token 错当成同一步的 query。EOS 也有相应记录。捕获次数与输出 token 数必须相等，否则报错。

默认色标在两个模型、全部步骤间共享（每个 layer/head 一组范围），展示原始局部权重。另有明确标记的 local contrast 模式，只增强每帧局部可见性，不能比较绝对强弱。HTML 和 NPZ 均保留 float32 权重，避免弱 attention 在显示量化时丢失。页面两条时间轴独立，因为相同 token index 不保证相同语义阶段。热图显示 patch 格子，不伪装成精确像素级定位；视觉特征已经融合上下文，attention 也不是因果归因。

生成文本在自然遇到 EOS 前达到 token 上限时，标记 `token_limit`，不会伪称 reasoning 完整。权重在生成中捕获，HTML 在两次生成完成后渲染；当前不提供边生成边刷新的服务端界面。

## 验证

```powershell
Set-Location VLM_adaptation/Evaluation_scripts/Attention_visualization
& C:/Users/junyhuang/.conda/envs/VisionGRPO/python.exe -m unittest test_attention.py
```

测试以真实 Qwen attention 模块核对 prefill、KV cache decode、head 顺序、网格重排及 hook 清理。运行环境需要 torch、transformers（支持 Qwen3-VL）、peft、accelerate、bitsandbytes、numpy、Pillow、matplotlib。测试过的具体环境和 GPU 运行结果见 `VALIDATION.md`。
