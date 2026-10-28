#!/usr/bin/env python3
"""Windows GUI probe: manually choose a cached nine-grid crop during greedy Qwen reasoning."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageTk

VLM_ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = VLM_ROOT.parent
for path in (VLM_ROOT / "SparseVLM_eliminateCONTEXT", VLM_ROOT / "Training_scripts" / "GRPO_lookbetter"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from rolling_sparse_qwen3vl import load_model
from data import load_mapverse_questions
from live_rollout import _append_prepared, _cached_vision, _commit, _prompt, _visual_module, prepare_crops
from live_rollout_direct_visual import append_prepared_native_visual_segment
from region_policy import nine_grid_boxes


def choose_grid(image: Image.Image, *, token_index: int, look_number: int, current_text: str,
                prior_choices: list[int]) -> int:
    """Show original image + row-major 3x3 grid and return one clicked action."""
    try:
        import tkinter as tk
        from tkinter import messagebox, scrolledtext
    except ImportError as exc:
        raise RuntimeError("This manual selector needs the standard Windows Tk installation.") from exc
    root = tk.Tk()
    root.title(f"Manual nine-grid LOOK {look_number} at reasoning token {token_index}")
    root.update_idletasks()
    max_width, max_height = max(720, root.winfo_screenwidth() - 620), max(540, root.winfo_screenheight() - 180)
    scale = min(1.0, max_width / image.width, max_height / image.height)
    shown = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS) if scale < 1 else image.copy()
    photo = ImageTk.PhotoImage(shown)
    outer = tk.Frame(root); outer.pack(fill="both", expand=True, padx=10, pady=10)
    left, right = tk.Frame(outer), tk.Frame(outer); left.pack(side="left", fill="both", expand=True); right.pack(side="right", fill="both", padx=(12,0))
    canvas = tk.Canvas(left, width=shown.width, height=shown.height, cursor="hand2", highlightthickness=0)
    canvas.pack(); canvas.create_image(0, 0, image=photo, anchor="nw")
    for action, (x0, y0, x1, y1) in enumerate(nine_grid_boxes(image.width, image.height)):
        dx0, dy0, dx1, dy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
        canvas.create_rectangle(dx0, dy0, dx1, dy1, outline="white", width=3, tags=(f"rectcell{action}",))
        canvas.create_text((dx0+dx1)/2, (dy0+dy1)/2, text=str(action), fill="white",
                           font=("Segoe UI", 20, "bold"), tags=(f"labelcell{action}",))
    tk.Label(right, text=f"LOOK {look_number}/9  |  token {token_index}\nClick one grid cell, then confirm.",
             justify="left", font=("Segoe UI", 11, "bold")).pack(anchor="w")
    tk.Label(right, text=f"Earlier selections: {prior_choices or 'none'}", justify="left").pack(anchor="w", pady=(6,6))
    text = scrolledtext.ScrolledText(right, width=55, height=30, wrap="word", font=("Consolas", 9))
    text.pack(fill="both", expand=True); text.insert("1.0", current_text); text.configure(state="disabled")
    selected: int | None = None
    status = tk.StringVar(value="No cell selected.")
    tk.Label(right, textvariable=status, justify="left").pack(anchor="w", pady=(8,4))

    def click(event):
        nonlocal selected
        col = min(2, max(0, int(event.x / shown.width * 3)))
        row = min(2, max(0, int(event.y / shown.height * 3)))
        selected = row * 3 + col
        for action in range(9):
            canvas.itemconfigure(f"rectcell{action}", outline=("#f59e0b" if action == selected else "white"), width=(5 if action == selected else 3))
        status.set(f"Selected action {selected}.  Click 'Append selected crop' to continue.")

    result: list[int] = []
    def accept():
        if selected is None:
            messagebox.showwarning("Select a region", "Click one of the nine cells first.")
            return
        result.append(selected); root.destroy()
    def cancel(): root.destroy()
    canvas.bind("<Button-1>", click)
    buttons = tk.Frame(right); buttons.pack(anchor="e", pady=(8,0))
    tk.Button(buttons, text="Cancel run", command=cancel).pack(side="right", padx=(6,0))
    tk.Button(buttons, text="Append selected crop", command=accept).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", cancel); root.mainloop()
    if not result:
        raise RuntimeError("Manual grid selection was cancelled; no further tokens were generated.")
    return result[0]


def append_prepared_with_assistant_replay(model, processor, original: Image.Image, prepared_entry, instruction: str,
                                          assistant_prefix: str, past, cursor: int, device, max_pixels: int):
    """Append cached crop features, then replay a short assistant suffix after the new user turn.

    The old cache is retained.  The replay is deliberate extra context that
    makes Qwen continue an assistant sentence rather than begin a fresh answer
    after every crop user turn.
    """
    from continuous_cache_qwen3vl import (_append_crop_then_assistant_replay, _resize_to_pixel_cap,
                                          _segment_positions)
    box, _, features = prepared_entry
    crop = _resize_to_pixel_cap(original.crop(box), max_pixels)
    inputs = _append_crop_then_assistant_replay(
        processor, [crop], instruction, assistant_prefix, device, max_pixels=max_pixels)
    positions = _segment_positions(model, inputs, cursor)
    mask = torch.ones((1, past.get_seq_length() + inputs["input_ids"].shape[1]), dtype=torch.long, device=device)
    kwargs = dict(inputs); kwargs.pop("attention_mask", None)
    with _cached_vision(_visual_module(model), features), torch.inference_mode():
        output = model(**kwargs, attention_mask=mask, position_ids=positions, past_key_values=past,
                       use_cache=True, return_dict=True, output_hidden_states=True)
    return output.logits[:, -1].argmax(dim=-1), output.past_key_values, int(positions.max().item()) + 1


def append_prepared_visual_embeddings(model, prepared_entry, past, cursor: int, device):
    """Append one native visual segment, without opening a chat turn."""
    pending, past, cursor, _ = append_prepared_native_visual_segment(
        model, prepared_entry, past, cursor, device, -1
    )
    return pending, past, cursor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=VLM_ROOT / "Datasets" / "When_Where_Crop" / "visual_easier_1000.json")
    parser.add_argument("--dataset-index", type=int, default=247, help="1-based unified JSON row; default is Russian-control city-count question.")
    parser.add_argument("--mapverse-root", type=Path, default=THESIS_ROOT / "MapVerse" / "data" / "imgs")
    parser.add_argument("--mapwise-root", type=Path, default=VLM_ROOT / "Datasets" / "mapwise-dataset")
    parser.add_argument("--adapter", type=Path, required=True); parser.add_argument("--model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=500); parser.add_argument("--look-every", type=int, default=20)
    parser.add_argument("--max-looks", type=int, default=9); parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--crop-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--crop-instruction", default="Continue the unfinished reasoning using this additional visual evidence.")
    parser.add_argument("--replay-tail-tokens", type=int, default=0,
                        help="After each crop user turn, replay this many assistant tokens to preserve local CoT continuity; 0 reproduces the original append protocol.")
    parser.add_argument("--append-mode", choices=("chat-turn", "visual-embeds"), default="chat-turn",
                        help="chat-turn reproduces the existing crop pipeline; visual-embeds appends merger tokens directly without a new chat turn.")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True); args = parser.parse_args()
    if args.dataset_index <= 0 or args.look_every <= 0 or args.max_looks < 0: parser.error("invalid selection schedule")
    if args.append_mode == "visual-embeds" and args.replay_tail_tokens:
        parser.error("--replay-tail-tokens applies only to --append-mode chat-turn")

    records = load_mapverse_questions(args.dataset, args.mapverse_root, mapwise_root=args.mapwise_root, limit=args.dataset_index)
    if len(records) < args.dataset_index: parser.error("dataset-index exceeds available questions")
    item = records[args.dataset_index - 1]
    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    with Image.open(item.image_path) as file: original = file.convert("RGB")
    print(f"Manual question: dataset row {args.dataset_index}, qa_id={item.sample_id}\n{item.question}", flush=True)
    print("Encoding all nine crop features once before generation...", flush=True)
    prepared = prepare_crops(model, processor, original, device, args.crop_max_pixels, args.crop_instruction)
    from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap, _segment_positions
    inputs = _content_inputs(processor, [_resize_to_pixel_cap(original, args.global_max_pixels)],
                             _prompt(item.question), device, args.global_max_pixels)
    positions = _segment_positions(model, inputs, 0)
    with torch.inference_mode():
        output = model(**inputs, position_ids=positions, use_cache=True, return_dict=True)
    pending, past, cursor = output.logits[:, -1].argmax(dim=-1), output.past_key_values, int(positions.max().item()) + 1
    eos = processor.tokenizer.eos_token_id; eos_ids = set(map(int, eos)) if isinstance(eos, (list,tuple,set)) else {int(eos)}
    ids: list[int] = []; choices: list[dict] = []; terminated = False
    for _ in range(args.max_new_tokens):
        token = int(pending.item()); ids.append(token)
        if token in eos_ids: terminated = True; break
        pending, past, cursor, _ = _commit(model, pending, past, cursor, -1)
        token_index = len(ids)
        if token_index % args.look_every == 0 and len(choices) < args.max_looks:
            current = processor.tokenizer.decode(ids, skip_special_tokens=True)
            print(f"\n=== PAUSE: LOOK {len(choices)+1} at token {token_index} ===\n{current}\n", flush=True)
            action = choose_grid(original, token_index=token_index, look_number=len(choices)+1,
                                 current_text=current, prior_choices=[entry["action"] for entry in choices])
            box = prepared[action][0]
            if args.append_mode == "visual-embeds":
                pending, past, cursor = append_prepared_visual_embeddings(model, prepared[action], past, cursor, device)
            elif args.replay_tail_tokens:
                tail = processor.tokenizer.decode(ids[-args.replay_tail_tokens:], skip_special_tokens=False)
                replay_instruction = ("The attached crop is fresh visual evidence for the assistant reasoning already in progress. "
                                      "Continue the exact unfinished reasoning in the assistant text that follows; do not restart the task or reintroduce the legend.")
                pending, past, cursor = append_prepared_with_assistant_replay(
                    model, processor, original, prepared[action], replay_instruction, tail, past, cursor, device,
                    args.crop_max_pixels)
            else:
                pending, past, cursor, _ = _append_prepared(model, prepared[action], past, cursor, device, -1)
            choices.append({"look":len(choices)+1, "reasoning_token_index":token_index, "action":action, "crop_box":box,
                            "cache_length_after_append":int(past.get_seq_length())})
            print(f"Appended action {action}, box={box}; cache length={past.get_seq_length()}.", flush=True)
    response = processor.tokenizer.decode(ids, skip_special_tokens=True)
    result={"dataset_index":args.dataset_index,"qa_id":item.sample_id,"image":str(item.image_path),"question":item.question,
            "manual_choices":choices,"response":response,"terminated":terminated,"final_cache_length":int(past.get_seq_length()),
            "cache_policy":"continuous_global_reasoning_and_all_manually_selected_crop_kv",
            "append_mode":args.append_mode,"replay_tail_tokens":args.replay_tail_tokens}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"\n=== FINAL RESPONSE ===\n{response}\n\nSaved: {args.output}", flush=True)

if __name__ == "__main__": main()
