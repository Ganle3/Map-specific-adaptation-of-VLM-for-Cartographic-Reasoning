"""Map-grouped Student examples backed by the existing Grad-Att collector files."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Example:
    map_id: str
    sample_id: str
    hidden_file: Path
    hidden_row: int
    token_index: int
    kv_file: Path
    target_files: tuple[Path, ...]
    target_tokens: int


def build_examples(source_root: Path, teacher_root: Path, k_max: int,
                   allow_missing_targets: bool = False) -> tuple[list[Example], dict]:
    """Join collector steps to encoded merger targets without losing empty steps.

    A resumed collector can append the same map/question again.  The latest
    manifest row for each (sample ID, map hash) wins, preventing duplicated
    reasoning steps from overweighting that question.
    """
    source_root, teacher_root = source_root.resolve(), teacher_root.resolve()
    manifest = source_root / "manifest.jsonl"
    index = teacher_root / "targets.jsonl"
    if not manifest.is_file() or not index.is_file():
        raise FileNotFoundError("Both the collector manifest and Teacher targets.jsonl are required.")
    target_index = {}
    for line in index.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            target_index[row["crop_path"]] = row
    rows = {}
    for line in manifest.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            rows[(str(row["sample_id"]), row["global_image"]["sha256_prefix"])] = row

    examples = []
    stats = {"manifest_rows_after_dedup": len(rows), "missing_targets": 0,
             "over_k_max": 0, "empty_steps": 0, "positive_steps": 0}
    for (_, map_id), row in rows.items():
        hidden_file = source_root / row["hidden_state_file"]
        kv_relative = row["global_image"]["visual_kv"]
        if kv_relative is None:
            raise ValueError(f"No persisted global visual KV for map {map_id}.")
        kv_file = source_root / kv_relative
        if not hidden_file.is_file() or not kv_file.is_file():
            raise FileNotFoundError(f"Missing hidden/KV file for map {map_id}.")
        for hidden_row, step in enumerate(row["steps"]):
            paths = [crop["path"] for crop in step["crops"]]
            missing = [path for path in paths if path not in target_index]
            if missing:
                stats["missing_targets"] += 1
                if allow_missing_targets:
                    continue
                raise FileNotFoundError(f"Teacher target missing for {missing[0]}; finish prepare_teacher.py first.")
            length = sum(int(target_index[path]["token_count"]) for path in paths)
            if length > k_max:
                stats["over_k_max"] += 1
                continue  # Never truncate a merger target or move its STOP label.
            files = tuple(teacher_root / target_index[path]["target_file"] for path in paths)
            if any(not path.is_file() for path in files):
                raise FileNotFoundError(f"Teacher target index points to a missing file: {files}")
            examples.append(Example(map_id, str(row["sample_id"]), hidden_file, hidden_row,
                                    int(step["token_index"]), kv_file, files, length))
            stats["positive_steps" if length else "empty_steps"] += 1
    if not examples:
        raise ValueError("No training examples remain after joining targets and applying k_max.")
    target_dims = {int(target_index[crop["path"]]["embedding_dim"])
                   for row in rows.values() for step in row["steps"] for crop in step["crops"]
                   if crop["path"] in target_index}
    if len(target_dims) != 1:
        raise ValueError(f"Expected one actual Qwen main-merger dimension, found {target_dims}.")
    stats["merger_dim"] = target_dims.pop()
    stats["unique_maps"] = len({example.map_id for example in examples})
    stats["usable_steps"] = len(examples)
    return examples, stats


def split_by_map(examples: list[Example], seed: int, validation_fraction: float,
                 test_fraction: float) -> dict[str, list[Example]]:
    """Never put steps or questions from the same image in different splits."""
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")
    maps = sorted({example.map_id for example in examples})
    if len(maps) < 3:
        raise ValueError("Map-level train/validation/test splits need at least three different images.")
    random.Random(seed).shuffle(maps)
    n_val = max(1, round(len(maps) * validation_fraction))
    n_test = max(1, round(len(maps) * test_fraction))
    if n_val + n_test >= len(maps):
        raise ValueError("Too few maps remain for the training split.")
    assignments = {map_id: "validation" for map_id in maps[:n_val]}
    assignments.update({map_id: "test" for map_id in maps[n_val:n_val + n_test]})
    return {name: [example for example in examples if assignments.get(example.map_id, "train") == name]
            for name in ("train", "validation", "test")}


class LatentCropDataset(Dataset):
    """Load one reasoning step; retain a small CPU LRU for repeated map data."""

    def __init__(self, examples: list[Example], merger_dim: int):
        self.examples = examples
        self.merger_dim = merger_dim

    def __len__(self) -> int:
        return len(self.examples)

    @lru_cache(maxsize=2)
    def _visual_kv(self, path: Path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        layer_ids = sorted(int(key.split("_")[1]) for key in payload if key.endswith("_key"))
        if layer_ids != list(range(len(layer_ids))):
            raise ValueError(f"Visual KV layer numbering is incomplete: {path}")
        keys, values = [], []
        for layer in layer_ids:
            stem = f"layer_{layer:02d}"
            for kind, output in (("key", keys), ("value", values)):
                tensor = payload[f"{stem}_{kind}"]
                if tensor.ndim != 4 or tensor.shape[0] != 1:
                    raise ValueError(f"Unexpected visual KV shape {tensor.shape} in {path}")
                # Cache order is [batch, GQA heads, visual positions, head dim].
                # Flatten heads only after moving positions ahead of heads.
                tensor = tensor[0].permute(1, 0, 2).reshape(tensor.shape[2], -1)
                if tensor.dtype == torch.int8:
                    scale = float(payload[f"{stem}_{kind}_scale"])
                    tensor = (tensor.float() * scale).to(torch.float16)
                else:
                    tensor = tensor.to(torch.float16)
                output.append(tensor)
        grid = torch.as_tensor(payload["metadata"]["image_grid"], dtype=torch.long)
        if len(grid) != 2 or keys[0].shape[0] != int(grid.prod()):
            raise ValueError(f"Saved visual token count does not match merged image grid in {path}.")
        return torch.stack(keys), torch.stack(values), grid

    @lru_cache(maxsize=4)
    def _hidden_states(self, path: Path):
        return torch.load(path, map_location="cpu", weights_only=True)

    @lru_cache(maxsize=64)
    def _target(self, path: Path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        embedding = payload["embedding"]
        grid = payload["image_grid_thw"]
        if embedding.ndim != 2 or len(grid) != 3:
            raise ValueError(f"Invalid Teacher merger target in {path}.")
        return embedding, grid

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        states = self._hidden_states(example.hidden_file)
        if int(states["token_indices"][example.hidden_row]) != example.token_index:
            raise ValueError(f"Hidden-state/manifest token mismatch for {example.sample_id}.")
        hidden = states["hidden_states_bf16"][example.hidden_row]
        keys, values, grid = self._visual_kv(example.kv_file)
        embeddings, crop_grids = [], []
        for target_file in example.target_files:
            embedding, crop_grid = self._target(target_file)
            embeddings.append(embedding)
            crop_grids.append(crop_grid)
        target = torch.cat(embeddings, dim=0) if embeddings else hidden.new_empty((0, self.merger_dim))
        if target.shape[0] != example.target_tokens:
            raise ValueError(f"Target length changed for {example.sample_id} token {example.token_index}.")
        return {"hidden": hidden, "keys": keys, "values": values, "global_grid": grid,
                "target": target, "length": example.target_tokens, "crop_grids": crop_grids,
                "map_id": example.map_id, "sample_id": example.sample_id,
                "token_index": example.token_index}


def collate_steps(samples: list[dict]) -> dict:
    """Pad only image-memory and target sequence axes; preserve map metadata."""
    batch = len(samples)
    layers, _, kv_dim = samples[0]["keys"].shape
    merger_dim = samples[0]["target"].shape[1]
    max_memory = max(sample["keys"].shape[1] for sample in samples)
    max_target = max(sample["length"] for sample in samples)
    keys = samples[0]["keys"].new_zeros((batch, layers, max_memory, kv_dim))
    values = torch.zeros_like(keys)
    target = samples[0]["hidden"].new_zeros((batch, max_target, merger_dim))
    memory_mask = torch.zeros((batch, max_memory), dtype=torch.bool)
    lengths = torch.tensor([sample["length"] for sample in samples], dtype=torch.long)
    for i, sample in enumerate(samples):
        if sample["keys"].shape[0] != layers or sample["keys"].shape[2] != kv_dim:
            raise ValueError("Mixed decoder KV architectures in one batch.")
        n = sample["keys"].shape[1]
        keys[i, :, :n] = sample["keys"]
        values[i, :, :n] = sample["values"]
        memory_mask[i, :n] = True
        if sample["length"]:
            if sample["target"].shape[1] != merger_dim:
                raise ValueError("Mixed Teacher merger dimensions in one batch.")
            target[i, :sample["length"]] = sample["target"]
    return {"hidden": torch.stack([sample["hidden"] for sample in samples]),
            "keys": keys, "values": values, "global_grid": torch.stack([sample["global_grid"] for sample in samples]),
            "memory_mask": memory_mask, "target": target, "lengths": lengths,
            "metadata": [{key: sample[key] for key in ("map_id", "sample_id", "token_index", "crop_grids")}
                         for sample in samples]}
