import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from tqdm import tqdm

from peft import LoraConfig


# --- 1. Configuration Class ---
@dataclass
class LoraMoEConfig(LoraConfig):
    num_experts: int = 4
    top_k: int = 2
    # Add other PEFT-like parameters if needed

    def __post_init__(self):
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})")


# --- 2. LoRA-MoE Layer Implementation ---
class LoraMoELinear(nn.Module):
    """
    A Linear layer with multiple LoRA experts and a gating mechanism.
    This is a simplified implementation.
    """
    def __init__(self, base_layer: nn.Linear, config: LoraMoEConfig):
        super().__init__()
        self.base_layer = base_layer
        self.r = config.r
        self.lora_alpha = config.lora_alpha
        self.lora_dropout = config.lora_dropout
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0

        # --- LoRA Experts ---
        # Create A and B matrices for each expert
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.empty(self.r, base_layer.in_features)) for _ in range(self.num_experts)]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.empty(base_layer.out_features, self.r)) for _ in range(self.num_experts)]
        )

        # --- Gate Network ---
        # Simple gate taking the input's mean across sequence (for demonstration)
        # In practice, the gate input might be different (e.g., from hidden states)
        self.gate = nn.Linear(base_layer.in_features, self.num_experts, bias=False)

        # --- Dropout ---
        if self.lora_dropout > 0:
            self.lora_dropout_layer = nn.Dropout(p=self.lora_dropout)
        else:
            self.lora_dropout_layer = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize LoRA A matrices (kaiming uniform) and B matrices (zeros)
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[i])
        # Initialize gate weights (optional, often defaults are fine)
        # nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Base Linear Forward ---
        base_output = self.base_layer(x)

        # --- Gate Calculation ---
        # Use the mean of the last sequence dimension for the gate input
        # This is a simplification. For attention layers (Q, K, V), you might use different logic.
        # Shape: (batch, seq_len, in_features) -> Gate Input: (batch, in_features)
        gate_input = x.mean(dim=-2) if x.dim() > 2 else x
        gate_logits = self.gate(gate_input) # Shape: (batch, num_experts)

        # --- Top-K Selection ---
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        # Normalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # --- Compute LoRA Contributions for Top-K Experts ---
        # This is the key part: efficiently compute contributions from selected experts
        # Method 1: Gather A/B matrices for selected experts and perform batched matmul
        # This is more efficient than looping.

        # Get selected A and B matrices for the batch
        # Shape of selected_A/B: (batch_size, top_k, ..._dim, in_features/r or out_features/r)
        selected_A = torch.stack([self.lora_A[expert_idx] for expert_idx in top_k_indices.view(-1)], dim=0).view(x.size(0), self.top_k, self.r, x.size(-1))
        selected_B = torch.stack([self.lora_B[expert_idx] for expert_idx in top_k_indices.view(-1)], dim=0).view(x.size(0), self.top_k, self.base_layer.out_features, self.r)

        # Expand x for batched matmul: (batch, seq, in) -> (batch, 1, 1, seq, in) -> (batch, top_k, 1, seq, in)
        x_expanded = x.unsqueeze(1).unsqueeze(1).expand(-1, self.top_k, -1, -1, -1) # (B, K, 1, S, I)
        selected_A_expanded = selected_A.unsqueeze(-2) # (B, K, R, 1, I)
        selected_B_expanded = selected_B.unsqueeze(-2).unsqueeze(-2) # (B, K, O, 1, 1)

        # Perform matmuls: (x @ A.T) @ B.T
        # First: x @ A.T -> (B, K, 1, S, I) @ (B, K, R, 1, I).T -> (B, K, 1, S, R)
        lora_delta_intermediate = torch.matmul(x_expanded, selected_A_expanded.transpose(-1, -2))
        # Second: (result) @ B.T -> (B, K, 1, S, R) @ (B, K, O, 1, 1).T -> (B, K, O, S, 1)
        lora_delta_full = torch.matmul(lora_delta_intermediate, selected_B_expanded.transpose(-1, -2))

        # Remove singleton dimensions: (B, K, O, S, 1) -> (B, K, O, S)
        lora_delta = lora_delta_full.squeeze(-1).transpose(-1, -2) # -> (B, K, S, O)

        # Apply scaling and gate weights
        # top_k_weights: (B, K) -> (B, K, 1, 1)
        lora_delta_weighted = lora_delta * top_k_weights.unsqueeze(-1).unsqueeze(-1) * self.scaling

        # Sum contributions from top-k experts: (B, K, S, O) -> (B, S, O)
        lora_output = lora_delta_weighted.sum(dim=1)

        # --- Combine Base and LoRA Output ---
        final_output = base_output + lora_output

        return final_output

# --- 3. Function to Add print_trainable_parameters Method ---
def _add_print_trainable_params_method(model):
    """
    Adds a print_trainable_parameters method to the given model object.
    """
    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    # Attach the method to the model instance
    # 'self' inside the method will refer to the model instance
    model.print_trainable_parameters = print_trainable_parameters.__get__(model, model.__class__)
    return model


# --- 3. Function to Modify Model ---
def modify_model_with_loramoe(model: nn.Module, config: LoraMoEConfig):
    """
    Modifies the given model in-place by replacing target Linear layers with LoraMoELinear.

    Args:
        model: The PyTorch model to modify.
        config: The LoraMoEConfig containing parameters.
    """
    for name, module in tqdm(model.named_modules(), desc='replacing model with LoraMoELinear'):
        # Check if the module is Linear and matches target modules
        if isinstance(module, nn.Linear) and any(target_name in name for target_name in config.target_modules):
            # Replace the module
            new_module = LoraMoELinear(module, config)
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_module)
            # print(f"Replaced {name} with LoraMoELinear")
    model = _add_print_trainable_params_method(model)
    return model
