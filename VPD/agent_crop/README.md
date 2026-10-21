# Qwen3-VL local image-zoom agent test

This folder contains an isolated experiment for the question:

> how many of the cities are in Russian military control?

using `2.webp`, the local `Qwen/Qwen3-VL-8B-Thinking` checkpoint, and the
existing PEFT adapter.  It uses the same repository loader as the consistency
experiments: 4-bit NF4 base model with BF16 compute, followed by PEFT LoRA
mounting, unless `--no-quantization` is supplied.

The default implementation follows AdaptVision's `request_local_region`
schema: `bbox_2d` is `[x1, y1, x2, y2]` in absolute pixels of the original
image, and the crop is taken from the original full-resolution image.  The
crop is saved as a lossless PNG without an extra resize and returned as a new
image content item in a `tool` message, not as a path in text.  Use
`--tool-name image_zoom_in_tool` to switch to Qwen-Agent's alternative tool
name.  The local checkpoint chat template supports the corresponding
`<tools>`, `<tool_call>`, and `<tool_response>` format.

Two modes are separated:

- `sanity`: explicitly asks for exactly one tool call;
- `autonomous`: exposes the tool but does not require using it.

The default `both` mode runs both with the same model, question, image, and
generation settings.  The default maximum is one tool call per mode.

Tool calls are now handled at generation time: a stopping criterion interrupts
the current generation immediately after `</tool_call>`, executes the crop, and
starts the next multimodal turn.  Therefore `--max-tool-calls 3` means up to
three genuinely interleaved crop/response steps during one reasoning trace,
not three crops prepared in advance.

## Run

```powershell
python .\VLM_adaptation\VPD\agent_crop\agent_crop.py `
  --mode both `
  --image "C:\Users\junyhuang\Thesis\MapVerse\data\imgs\2.webp" `
  --adapter-path "C:\Users\junyhuang\Thesis\VLM_adaptation\Training_outputs\MapVerse_scaling1000_GRPO_from_MapVerse5000_token500\checkpoint-2500" `
  --tool-name request_local_region `
  --output-dir ".\VLM_adaptation\VPD\agent_crop\outputs_checkpoint2500"
```

Base-only test:

```powershell
python .\VLM_adaptation\VPD\agent_crop\agent_crop.py `
  --mode both `
  --no-adapter `
  --output-dir ".\VLM_adaptation\VPD\agent_crop\outputs_base"
```

Each run writes `agent_crop_results.json`, complete raw turns, parsed tool
calls, validation/crop errors, final answer, timing, and crop PNGs under
`sanity\crops` or `autonomous\crops`.

The implementation references Qwen-Agent's official files:

- `qwen_agent/tools/image_zoom_in_qwen3vl.py`
- `examples/assistant_qwen3vl.py`

It does not import the cloud-only `qwenvl_dashscope` agent backend.
