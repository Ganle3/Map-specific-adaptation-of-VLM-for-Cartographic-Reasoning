"""Question-conditioned prototype queries followed by image cross-attention.

Only ``prototypes`` is trainable in the default experiment::

    C = CrossAttention_question(Q=P, K=H_question, V=H_question)
    Q_conditioned = LayerNorm(P + C)
    R = CrossAttention_image(Q=Q_conditioned, K=Z_image, V=Z_image)

``H_question`` comes from the frozen Qwen final decoder layer on the original
image+question prompt.  Both attention modules use frozen identity projections.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QuestionConditionedPrototypePolicy(nn.Module):
    def __init__(self, hidden_size: int, *, num_prototypes: int = 9,
                 num_heads: int = 8, train_projections: bool = False):
        super().__init__()
        if hidden_size <= 0 or num_prototypes <= 0 or hidden_size % num_heads:
            raise ValueError("Invalid hidden size, prototype count, or attention head count")
        self.hidden_size = int(hidden_size)
        self.num_prototypes = int(num_prototypes)
        self.num_heads = int(num_heads)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_size) * 0.02)
        self.question_attention = nn.MultiheadAttention(
            hidden_size, num_heads, bias=False, batch_first=True
        )
        self.image_attention = nn.MultiheadAttention(
            hidden_size, num_heads, bias=False, batch_first=True
        )
        self._initialize_identity(self.question_attention)
        self._initialize_identity(self.image_attention)
        for module in (self.question_attention, self.image_attention):
            for parameter in module.parameters():
                parameter.requires_grad_(train_projections)

    @torch.no_grad()
    def _initialize_identity(self, module: nn.MultiheadAttention) -> None:
        eye = torch.eye(self.hidden_size)
        module.in_proj_weight.copy_(torch.cat((eye, eye, eye), dim=0))
        module.out_proj.weight.copy_(eye)

    def forward(self, image_tokens: torch.Tensor,
                question_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nine image-memory tokens and their per-head image attention."""
        if image_tokens.ndim != 2 or image_tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected image_tokens [N,{self.hidden_size}]")
        if question_tokens.ndim != 2 or question_tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected question_tokens [M,{self.hidden_size}]")
        prototypes = F.layer_norm(self.prototypes, (self.hidden_size,)).unsqueeze(0)
        question = F.layer_norm(question_tokens.float(), (self.hidden_size,)).unsqueeze(0)
        question_context, question_weights = self.question_attention(
            prototypes, question, question, need_weights=True, average_attn_weights=False
        )
        conditioned_query = F.layer_norm(
            prototypes + question_context, (self.hidden_size,)
        )
        image_key = F.layer_norm(image_tokens.float(), (self.hidden_size,)).unsqueeze(0)
        image_value = image_tokens.float().unsqueeze(0)
        memory, image_weights = self.image_attention(
            conditioned_query, image_key, image_value,
            need_weights=True, average_attn_weights=False,
        )
        # Diagnostic only; detached so the module never retains a training graph.
        self.last_question_attention = question_weights.detach()
        return memory[0], image_weights

    def trainable_parameters(self) -> list[nn.Parameter]:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if len(parameters) != 1 or parameters[0] is not self.prototypes:
            raise RuntimeError("Default experiment must optimize only policy.prototypes")
        return parameters
