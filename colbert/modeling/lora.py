import torch
import torch.nn as nn


class ExpertLoRALinear(nn.Module):
    def __init__(self, base_linear, num_experts, rank, alpha=16.0, dropout=0.0):
        super().__init__()

        self.base = base_linear
        self.num_experts = num_experts
        self.rank = rank
        self.scaling = alpha / max(1, rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._gate_weights = None

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        if rank > 0 and num_experts > 0:
            self.lora_A = nn.Parameter(torch.zeros(num_experts, rank, base_linear.in_features))
            self.lora_B = nn.Parameter(torch.zeros(num_experts, base_linear.out_features, rank))

            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    def set_gate_weights(self, gate_weights):
        self._gate_weights = gate_weights

    def clear_gate_weights(self):
        self._gate_weights = None

    def forward(self, x):
        output = self.base(x)

        if self.rank <= 0 or self._gate_weights is None:
            return output

        gate_weights = self._gate_weights
        if gate_weights.dim() != 2 or gate_weights.size(0) != x.size(0):
            raise ValueError("Gate weights must be shaped as [batch, num_experts].")

        dropped = self.dropout(x)

        if dropped.dim() == 2:
            dropped = dropped.unsqueeze(1)
            output = output.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        delta = torch.zeros_like(output)
        for expert_idx in range(self.num_experts):
            expert_gate = gate_weights[:, expert_idx].view(-1, 1, 1)
            if torch.count_nonzero(expert_gate.detach()) == 0:
                continue

            low_rank = torch.einsum('bsi,ri->bsr', dropped, self.lora_A[expert_idx])
            expert_delta = torch.einsum('bsr,or->bso', low_rank, self.lora_B[expert_idx])
            delta = delta + expert_delta * (expert_gate * self.scaling)

        output = output + delta
        return output.squeeze(1) if squeeze_output else output


def inject_expert_lora(module, target_suffixes, num_experts, rank, alpha=16.0, dropout=0.0):
    for child_name, child in list(module.named_children()):
        replaced = False
        if isinstance(child, nn.Linear):
            for suffix in target_suffixes:
                if child_name == suffix or suffix.endswith(child_name):
                    setattr(module, child_name, ExpertLoRALinear(
                        child,
                        num_experts=num_experts,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    ))
                    replaced = True
                    break

        if not replaced:
            inject_expert_lora(child, target_suffixes, num_experts, rank, alpha=alpha, dropout=dropout)


def iter_expert_lora_modules(module):
    for child in module.modules():
        if isinstance(child, ExpertLoRALinear):
            yield child
