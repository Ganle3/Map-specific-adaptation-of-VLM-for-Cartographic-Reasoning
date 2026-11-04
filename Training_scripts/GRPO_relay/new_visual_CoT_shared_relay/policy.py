"""Parameter-light relay policy for coarse localization and local visual reading."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NewVisualCoTRelay(nn.Module):
    """Shared learnable relays used by both visual reads and language fusion.

    The demo deliberately has no H-by-H projection matrices.  Its only
    trainable tensor is ``prototypes`` with shape ``[num_relays, hidden_size]``.
    Normalized dot products keep the untrained random initialization numerically
    controlled while preserving a direct gradient path for a later GRPO trainer.
    """

    def __init__(self, hidden_size: int, *, num_relays: int = 2,
                 init_std: float = 0.02, temperature: float = 1.0,
                 injection_scale: float = 0.1) -> None:
        super().__init__()
        if hidden_size <= 0 or num_relays <= 0:
            raise ValueError("hidden_size and num_relays must be positive")
        if init_std <= 0 or temperature <= 0 or injection_scale < 0:
            raise ValueError("invalid initialization, temperature, or injection scale")
        self.hidden_size = int(hidden_size)
        self.num_relays = int(num_relays)
        self.temperature = float(temperature)
        self.injection_scale = float(injection_scale)
        self.prototypes = nn.Parameter(
            torch.randn(self.num_relays, self.hidden_size) * float(init_std)
        )

    @staticmethod
    def _normalized(value: torch.Tensor) -> torch.Tensor:
        # Calculate similarities in fp32.  NF4 model activations are normally
        # bf16, whose precision is unnecessarily low for diagnostic attention.
        return F.normalize(value.float(), dim=-1)

    def conditioned_queries(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Condition every shared relay on the current layer hidden state."""
        if hidden_state.ndim != 3 or hidden_state.shape[1] != 1:
            raise ValueError("hidden_state must have shape [batch, 1, hidden_size]")
        if hidden_state.shape[-1] != self.hidden_size:
            raise ValueError("hidden state and relay hidden sizes differ")
        state = self._normalized(hidden_state)
        prototypes = self._normalized(self.prototypes)[None, :, :]
        return self._normalized(state + prototypes)

    def _read(self, queries: torch.Tensor, memory: torch.Tensor):
        if memory.ndim == 2:
            memory = memory.unsqueeze(0)
        if queries.ndim != 3 or memory.ndim != 3:
            raise ValueError("queries and memory must be rank-3 tensors")
        if queries.shape[0] != memory.shape[0] or memory.shape[-1] != self.hidden_size:
            raise ValueError("query/memory batch or hidden-size mismatch")
        keys = self._normalized(memory)
        # Queries and keys are already unit-normalized, so this is cosine
        # attention.  Dividing again by sqrt(H) would make thousands of patch
        # probabilities almost exactly uniform and starve the selector gradient.
        logits = torch.matmul(queries, keys.transpose(-2, -1)) / self.temperature
        probabilities = torch.softmax(logits, dim=-1)
        values = memory.float()
        readout = torch.matmul(probabilities, values)
        return readout, logits, probabilities

    def select_global_patches(self, hidden_state: torch.Tensor,
                              global_memory: torch.Tensor, *, sample: bool = False):
        """Return one coarse merged-patch index per shared relay."""
        queries = self.conditioned_queries(hidden_state)
        _, relay_logits, relay_probabilities = self._read(queries, global_memory)
        distribution = torch.distributions.Categorical(logits=relay_logits)
        actions = distribution.sample() if sample else relay_logits.argmax(dim=-1)
        return {
            "actions": actions,
            "selected_log_probs": distribution.log_prob(actions),
            "patch_logits": relay_logits,
            "patch_probabilities": relay_probabilities,
        }

    def read_assigned_local_memories(self, hidden_state: torch.Tensor,
                                     local_memories: list[torch.Tensor]):
        """Let relay i read only the HR tile selected by relay i."""
        if len(local_memories) != self.num_relays:
            raise ValueError("Expected exactly one local memory per relay")
        queries = self.conditioned_queries(hidden_state)
        readouts, logits, probabilities = [], [], []
        for relay_index, memory in enumerate(local_memories):
            query = queries[:, relay_index:relay_index + 1]
            readout, relay_logits, relay_probs = self._read(query, memory)
            readouts.append(readout)
            logits.append(relay_logits)
            probabilities.append(relay_probs)
        return (
            torch.cat(readouts, dim=1).to(hidden_state.dtype),
            logits,
            probabilities,
        )

    def fuse_hidden_state(self, hidden_state: torch.Tensor,
                          visual_relays: torch.Tensor):
        """Let the language state query the visual relays, without adding KV."""
        if visual_relays.ndim != 3 or visual_relays.shape[1] != self.num_relays:
            raise ValueError("visual_relays must have shape [batch, num_relays, H]")
        query = self._normalized(hidden_state)
        keys = self._normalized(visual_relays)
        logits = torch.matmul(query, keys.transpose(-2, -1)) / self.temperature
        probabilities = torch.softmax(logits, dim=-1)
        readout = torch.matmul(probabilities, visual_relays.float()).to(hidden_state.dtype)
        fused = hidden_state + self.injection_scale * readout
        return fused, logits, probabilities
