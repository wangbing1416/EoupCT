import torch
import logging
from build_trainer import CLTrainer

logger = logging.getLogger(__name__)


class RoupCTTrainer(CLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.M_old = []  # historical gradient memory
        self.alpha = getattr(self.args, "roupct_alpha", 0.05)
        self.lambda_pen = getattr(self.args, "roupct_lambda_pen", 0.1)
        self.seq_len = getattr(self.args, "roupct_seq_len", 16)
        self.soft_prompt_lr = getattr(self.args, "roupct_soft_prompt_lr", 1e-3)
        self.soft_P = None
        self.P_opt = None
        self.embed_layer = None
        self.tau = getattr(self.args, "roupct_tau", 0.5)
        
    def _get_lora_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _get_reference_tensor(self, model):
        params = list(model.parameters())
        if len(params) == 0:
            raise RuntimeError("RoupCTTrainer requires model parameters to infer device/dtype")
        return params[0]
        
    def _flatten_grads(self, grads):
        return torch.cat([g.reshape(-1) for g in grads])
        
    def _unflatten_grads(self, flat_grad, params):
        grads = []
        offset = 0
        for p in params:
            numel = p.numel()
            grads.append(flat_grad[offset:offset+numel].view(p.shape))
            offset += numel
        return grads

    def _generate_soft_sequence(self, P):
        # Optimization 1: Bypass autoregressive generation by directly 
        # using the prompt P (initialized to length seq_len) as the continuous surrogate sequence.
        # This reduces 16 forward passes to 0 during sequence generation.
        return P
        
    def _compute_kd_loss(self, S_soft):
        # We need the pure base model output and the LoRA-adapted output.
        # In peft, we can temporarily disable LoRA adapters explicitly.
        with getattr(self.model, "disable_adapter", lambda: torch.no_grad())():
            with torch.no_grad():
                base_outputs = self.model(inputs_embeds=S_soft)
                base_logits = base_outputs.logits.detach()
                
        lora_outputs = self.model(inputs_embeds=S_soft)
        lora_logits = lora_outputs.logits
        return torch.nn.functional.mse_loss(lora_logits, base_logits)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        lora_params = self._get_lora_params()
        if self.soft_P is None:
            reference_tensor = self._get_reference_tensor(model)
            self.soft_P = torch.nn.Parameter(
                torch.randn(
                    1,
                    self.seq_len,
                    self.model.config.hidden_size,
                    device=reference_tensor.device,
                    dtype=reference_tensor.dtype,
                ) * 0.02
            )
            self.P_opt = torch.optim.Adam([self.soft_P], lr=self.soft_prompt_lr)

        # 1. Compute task specific gradient g_new
        loss_new = self.compute_loss(model, inputs)
        g_new_tuple = torch.autograd.grad(loss_new, lora_params, retain_graph=False)
        g_new = self._flatten_grads(g_new_tuple).detach()

        # 2. Virtual step approximation
        original_weights = []
        for p, g in zip(lora_params, g_new_tuple):
            original_weights.append(p.data.clone())
            with torch.no_grad():
                p.add_(-g * self.alpha)
            
        # 3. Optimize soft prompt P* at virtual weights to target vulnerabilities
        self.P_opt.zero_grad()
        S_soft_virtual = self._generate_soft_sequence(self.soft_P)
        loss_PT_virtual = self._compute_kd_loss(S_soft_virtual)
        
        # Optimization 2: only optimize the soft prompt here.
        # Avoid calling backward() on model parameters, which can trigger DeepSpeed ZeRO
        # gradient reduction for params that are not participating in this auxiliary step.
        adv_loss = -loss_PT_virtual  # purely maximize vulnerability
        soft_prompt_grad = torch.autograd.grad(adv_loss, self.soft_P, retain_graph=False, allow_unused=False)[0]
        self.soft_P.grad = soft_prompt_grad.detach()
        self.P_opt.step()
        
        # Restore pre-virtual weights
        for p, orig_w in zip(lora_params, original_weights):
            with torch.no_grad():
                p.copy_(orig_w)
            
        # 4. Compute optimal protection gradient g_PT* at real weights
        S_soft_star = self._generate_soft_sequence(self.soft_P)
        loss_PT_star = self._compute_kd_loss(S_soft_star)
        g_PT_tuple = torch.autograd.grad(loss_PT_star, lora_params)
        g_PT = self._flatten_grads(g_PT_tuple).detach()
        
        # 5. Null space projection using historical gradients M_old
        g_new_tilde = g_new
        g_PT_tilde = g_PT
        if len(self.M_old) > 0:
            # Gram-Schmidt or projection matrix
            proj_dtype = torch.float32
            M = torch.stack([m.to(dtype=proj_dtype) for m in self.M_old]).t()  # [num_params, N]
            try:
                # Efficient projection to avoid OOM: g - M @ (M^T M)^-1 @ (M^T @ g)
                M_T = M.t()
                eye = torch.eye(M.shape[1], device=M.device, dtype=proj_dtype)
                inv_term = torch.linalg.inv(M_T @ M + 1e-4 * eye)
                
                # For g_new
                g_new_proj = g_new.to(dtype=proj_dtype)
                alpha_new = inv_term @ (M_T @ g_new_proj)
                g_new_tilde = g_new_proj - M @ alpha_new
                
                # For g_PT
                g_PT_proj = g_PT.to(dtype=proj_dtype)
                alpha_PT = inv_term @ (M_T @ g_PT_proj)
                g_PT_tilde = g_PT_proj - M @ alpha_PT
            except Exception as e:
                logger.warning(f"Orthogonal projection failed: {e}")
                pass
                
        # 6. Pareto-optimal conflict resolution
        score = torch.dot(g_new_tilde, g_PT_tilde)
        if score < 0:
            norm_sq = torch.dot(g_PT_tilde, g_PT_tilde) + 1e-8
            g_new_star = g_new_tilde - (score / norm_sq) * g_PT_tilde
        else:
            g_new_star = g_new_tilde
            
        # Final update rule combination:
        final_grad = g_new_star + g_PT_tilde
        
        # Scale for gradient accumulation
        accum_steps = getattr(self.args, "gradient_accumulation_steps", 1)
        final_grad = final_grad / accum_steps
        final_grad = final_grad.to(dtype=lora_params[0].dtype)

        # Assign back to parameters so HF optimizer can step
        grads_to_apply = self._unflatten_grads(final_grad, lora_params)
        
        if len(lora_params) > 0:
            surrogate_loss = 0.0
            main_device = lora_params[0].device
            for p, g in zip(lora_params, grads_to_apply):
                # Construct surrogate loss to cleanly trigger DP/DeepSpeed backward hooks & accumulation
                surrogate_loss = surrogate_loss + torch.sum(p * g.detach().to(p.device)).to(main_device)
                
            if hasattr(self, "accelerator") and self.accelerator is not None:
                self.accelerator.backward(surrogate_loss)
            else:
                surrogate_loss.backward()
            
        return loss_new.detach()

