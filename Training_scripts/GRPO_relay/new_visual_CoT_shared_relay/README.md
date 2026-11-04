# Relay 4 lazy high-resolution demo

This inference-only demo keeps Qwen3-VL's normal one-million-pixel global
prefill. No high-resolution tile is processed during prefill or the first ten
generated reasoning tokens.

After generated token 10, each shared randomly initialized relay uses the
current decoder-layer hidden state to select its own coarse global merger token.
Each selected window is mapped to original-image coordinates and encoded on
demand. If relays select the same window, only the first request runs the ViT.
Merger embeddings are cached on CPU. Local visual features never enter the
language sequence or persistent KV cache: every relay reads its assigned local
memory and the current hidden state reads the compact relay results before the
next layer.

Run question 318 (`764.png`, expected answer `Rastan`) from the Thesis root:

```powershell
.\VLM_adaptation\Training_scripts\GRPO_relay\new_visual_CoT_shared_relay\run_demo_qa318.ps1
```

Artifacts are written below:

```text
VLM_adaptation/Training_outputs/GRPO_relay/relay4_demo_qa318/
  random_relay_initialization.pt
  qa318_relay4_demo.json
  selected_tiles/*.png
```

The first lookup while processing token 10 affects prediction from token 11.
The JSON trace records each selected global patch, original bbox, crop path,
cache hit/miss, attention diagnostics, and reasoning generated so far.
