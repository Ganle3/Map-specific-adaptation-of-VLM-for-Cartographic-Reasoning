# Compatibility boundary: preserve HOW, replace observation policy

The reference implementation is `continuous_cache_qwen3vl.py`, but only its
validated **HOW** mechanism is frozen for this experiment. No controller code
may redesign or duplicate the following operations:

1. build the immutable full-image + question anchor KV;
2. crop the original high-resolution image and apply the existing per-crop
   pixel-budget code;
3. encode selected crop images through the existing Qwen3-VL processor/vision
   path;
4. append them as a multimodal user turn;
5. append the latest 600 reasoning tokens after the assistant header as actual
   assistant tokens; and
6. use the existing M-RoPE positions, cache recomputation, and continuation
   behavior to generate the next reasoning token.

The controller replaces the old observation policy completely. It does **not**
inherit mandatory fixed windows, 3x3 bootstrap coverage, full-image scout,
novelty filtering, fixed heuristic region retention, or any window-indexed
trigger rule.

## Required compatibility tests

### HOW insertion equivalence

For a fixed image, question, externally supplied crop bboxes, and supplied
600-token reasoning suffix, compare the new controller adapter against a direct
call to the existing `_append_crop_then_assistant_replay` + `_generate_segment`
path. The following must agree: input IDs, image grids, crop placeholder
positions, M-RoPE positions, cache length, generated token IDs, and output
text. This test has no policy decisions.

### Continue equivalence

With no LOOK action, the controller must run the normal decoder one token at a
time. Its concatenated token IDs must equal an ordinary greedy generation from
the same cache state. No crop/replay/user turn is inserted on CONTINUE.

### Trajectory provenance

Every rollout must save each LOOK boundary, selected source-coordinate bboxes,
submitted crop pixel sizes, replay suffix token count, cache lengths, generated
token IDs, and all policy actions/log-probabilities.

## Optional attention modulation

`global_key_bias`, `global_retain_attention_mass`, and `crop_key_bias` are
explicit controller-run configuration fields. They are not prerequisites of
crop insertion/replay. A run record must state whether each was active, its
numeric value, which original global visual key positions were suppressed, and
which current-crop key positions were boosted. No controller mode silently
changes their values or enables them based on an old window index.

## Adaptive execution

```text
continuous reasoning token
  -> WHEN: CONTINUE
       -> next reasoning token
  -> WHEN: LOOK
       -> WHERE: attribution proposals -> keep/drop -> legal size sample
       -> existing crop insertion + assistant replay + KV refresh
       -> continue unfinished reasoning
```

The frozen heuristic attribution machinery can provide WHERE candidates, but
the controller—not bootstrap/scout/novelty schedule—decides whether, when, and
which candidates are executed.
