# Relay1 step-4000 attention visualization

This captures the two fixed visual relay tokens during the single decoder
prefill. Each raw array has axes `[relay_token, decoder_layer, patch_y,
patch_x]`. It does not add an inference-step axis because Relay1 is not
recomputed during autoregressive decoding.

Run locally:

```powershell
.\VLM_adaptation\Training_scripts\GRPO_relay\relay1_visualization\run_step4000.ps1
```

Outputs are written to
`VLM_adaptation/Training_outputs/Prototype/relay1_visualization_step4000`.
The HTML viewer keeps R1 and R2 separate and can play their layer-by-layer
prefill attention.
