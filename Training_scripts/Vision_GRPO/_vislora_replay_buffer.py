"""Bounded CPU replay storage; no training dependencies, no global RNG changes."""
import hashlib
import json
import math


def prompt_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


class SuccessBuffer:
    def __init__(self, capacity=2):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.items = {}
        self.cursor = {}

    def add(self, qa_id, prompt_ids, tokens, logps, reward, terminated, step):
        if reward != 1.0 or not terminated:
            return False
        if not tokens or len(tokens) != len(logps) or not all(math.isfinite(x) for x in logps):
            raise ValueError("Invalid replay token/log-probability alignment")
        entry = dict(prompt_hash=prompt_digest(prompt_ids), tokens=list(tokens),
                     logps=list(logps), reward=1.0, terminated=True, step=int(step))
        bucket = self.items.setdefault(qa_id, [])
        # Refresh a duplicate's policy provenance; otherwise retain newest distinct successes.
        bucket[:] = [x for x in bucket if x["tokens"] != entry["tokens"]]
        bucket.append(entry)
        del bucket[:-self.capacity]
        return True

    def choose(self, qa_id, prompt_ids, step, max_age):
        valid = [x for x in self.items.get(qa_id, [])
                 if x["prompt_hash"] == prompt_digest(prompt_ids)
                 and 0 < step - x["step"] <= max_age]
        if not valid:
            return None
        cursor = self.cursor.get(qa_id, 0)
        self.cursor[qa_id] = cursor + 1
        return valid[cursor % len(valid)]

    def state_dict(self):
        return dict(capacity=self.capacity, items=self.items, cursor=self.cursor)
