import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os


class GPT2ROMEEditor:
    def __init__(self, model_name="gpt2-xl", device="cuda", alpha=0.1, edit_layers=2):
        self.device = device
        self.alpha = alpha
        self.edit_layers = edit_layers

        print(f"Loading model {model_name} on {device}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model loaded.")

    def _get_activations_at_position(self, input_ids, attention_mask, target_pos, module, layer_idx=None):
        """Get activations at a specific token position"""
        out, inp = {}, {}

        def forward_hook(module_obj, input, output):
            # Get activation at target position
            inp['x'] = input[0][:, target_pos, :].detach()

        def output_hook(module_obj, input, output):
            out['y'] = output[:, target_pos, :].detach()

        handles = []
        if module == 'mlp':
            layer = self.model.transformer.h[layer_idx].mlp.c_fc
            handles.append(layer.register_forward_hook(forward_hook))
            handles.append(layer.register_forward_hook(output_hook))
        else:  # For hidden states
            def hook_hidden(module_obj, input, output):
                hidden = output.hidden_states[-1][:, target_pos, :]
                out['y'] = hidden.detach()
                inp['x'] = hidden.detach()

            handles.append(self.model.register_forward_hook(hook_hidden))

        with torch.no_grad():
            _ = self.model(input_ids,
                           attention_mask=attention_mask,
                           output_hidden_states=True)

        for h in handles:
            h.remove()

        return out.get('y'), inp.get('x')

    def _compute_rank1_update(self, old_out, new_out, inp):
        """Compute rank-1 update matrix"""
        if old_out is None or new_out is None or inp is None:
            raise RuntimeError("Missing activation data")

        # Flatten batch dimension if present
        if old_out.dim() > 1:
            old_out = old_out.mean(dim=0)
        if new_out.dim() > 1:
            new_out = new_out.mean(dim=0)
        if inp.dim() > 1:
            inp = inp.mean(dim=0)

        delta = new_out - old_out

        # Normalize input to prevent numerical issues
        inp_norm = inp / (inp.norm() + 1e-8)

        # Compute outer product for rank-1 update
        u = delta
        v = inp_norm

        return u, v

    def _apply_rank1_update(self, weight, u, v, alpha):
        """Apply rank-1 update to weight matrix"""
        # Compute outer product update
        update = alpha * torch.outer(u, v)

        # Ensure dimensions match
        if update.shape != weight.data.shape:
            if update.shape == weight.data.shape[::-1]:
                update = update.t()
            else:
                raise RuntimeError(f"Update shape {update.shape} doesn't match weight shape {weight.data.shape}")

        weight.data += update

    def find_subject_last_token_pos(self, prompt_text):
        """Find the position of the last token of the subject in the prompt"""
        tokens = self.tokenizer.encode(prompt_text)
        # Return position of last token (where we expect the answer to follow)
        return len(tokens) - 1

    def edit_fact(self, subject, relation, old_obj, new_obj, layers=None):
        """Edit a factual association in the model"""

        # Create properly formatted prompts
        prompt = f"{subject}{relation}"  # e.g., "Danielle Darrieux's mother tongue is"
        old_completion = f"{prompt} {old_obj}"
        new_completion = f"{prompt} {new_obj}"

        print(f"Editing: '{prompt}' from '{old_obj}' to '{new_obj}'")

        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        old_ids = self.tokenizer.encode(old_completion, return_tensors='pt').to(self.device)
        new_ids = self.tokenizer.encode(new_completion, return_tensors='pt').to(self.device)

        # Find the position where we want to intervene (last token of prompt)
        target_pos = prompt_ids.shape[1] - 1

        # Auto-select layers if not provided
        if layers is None:
            layers = self.find_best_layers(old_ids, new_ids, target_pos)

        print(f"Applying updates to layers: {layers}")

        # Apply MLP updates
        for layer_idx in layers:
            try:
                old_out, old_inp = self._get_activations_at_position(
                    old_ids, torch.ones_like(old_ids), target_pos, 'mlp', layer_idx
                )
                new_out, new_inp = self._get_activations_at_position(
                    new_ids, torch.ones_like(new_ids), target_pos, 'mlp', layer_idx
                )

                if old_out is not None and new_out is not None and old_inp is not None:
                    u, v = self._compute_rank1_update(old_out, new_out, (old_inp + new_inp) / 2)

                    # Apply update to MLP layer
                    mlp_weight = self.model.transformer.h[layer_idx].mlp.c_fc.weight
                    self._apply_rank1_update(mlp_weight, u, v, self.alpha)
                    print(f"Applied MLP update to layer {layer_idx}")

            except Exception as e:
                print(f"Failed to update layer {layer_idx}: {e}")
                continue

        # Gentle LM head adjustment
        self._adjust_lm_head(prompt, old_obj, new_obj)

    def _adjust_lm_head(self, prompt, old_obj, new_obj):
        """Balanced LM head adjustment - strong enough to work but not cause loops"""
        # Get token IDs for old and new objects (with proper spacing)
        old_tokens = self.tokenizer.encode(f" {old_obj}", add_special_tokens=False)
        new_tokens = self.tokenizer.encode(f" {new_obj}", add_special_tokens=False)

        print(f"Old tokens: {old_tokens} ({[self.tokenizer.decode([t]) for t in old_tokens]})")
        print(f"New tokens: {new_tokens} ({[self.tokenizer.decode([t]) for t in new_tokens]})")

        with torch.no_grad():
            # Get baseline hidden state from the prompt
            prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            outputs = self.model(prompt_ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # Last token's hidden state

            # Normalize the hidden state
            hidden_norm = last_hidden / (last_hidden.norm() + 1e-8)

            # Balanced adjustment - enough to change preference but not dominate everything
            lm_alpha = self.alpha * 0.5  # Much more conservative

            # Apply direct bias to the logit weights
            for token_id in new_tokens:
                # Boost new object tokens moderately
                self.model.lm_head.weight.data[token_id] += lm_alpha * hidden_norm.squeeze()
                print(f"Boosted token {token_id} ({self.tokenizer.decode([token_id])})")

            for token_id in old_tokens:
                # Suppress old object tokens moderately
                self.model.lm_head.weight.data[token_id] -= lm_alpha * hidden_norm.squeeze()
                print(f"Suppressed token {token_id} ({self.tokenizer.decode([token_id])})")

        print(f"Applied LM head adjustments with alpha={lm_alpha}")

    def find_best_layers(self, old_ids, new_ids, target_pos, top_k=None):
        """Find layers with the highest activation differences"""
        if top_k is None:
            top_k = self.edit_layers

        layer_impacts = {}

        for layer_idx in range(len(self.model.transformer.h)):
            try:
                old_out, _ = self._get_activations_at_position(
                    old_ids, torch.ones_like(old_ids), target_pos, 'mlp', layer_idx
                )
                new_out, _ = self._get_activations_at_position(
                    new_ids, torch.ones_like(new_ids), target_pos, 'mlp', layer_idx
                )

                if old_out is not None and new_out is not None:
                    diff = (new_out - old_out).norm().item()
                    layer_impacts[layer_idx] = diff

            except Exception:
                continue

        # Return top-k layers with highest impact
        sorted_layers = sorted(layer_impacts.keys(), key=lambda x: layer_impacts[x], reverse=True)
        return sorted_layers[:top_k]

    def generate_text(self, prompt, max_length=50):
        """Generate text deterministically for easier debugging"""
        enc = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                enc.input_ids,
                attention_mask=enc.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_length=max_length,
                do_sample=False,
            )

        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.model = GPT2LMHeadModel.from_pretrained(path).to(self.device)
        self.model.eval()



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Increase alpha for more noticeable effects during debugging
    editor = GPT2ROMEEditor(device=device, alpha=0.15, edit_layers=2)

    prompt = "Danielle Darrieux's mother tongue is"
    print("Before:", editor.generate_text(prompt))

    # Edit the fact
    editor.edit_fact("Danielle Darrieux", "'s mother tongue is", "French", "English")

    print("After:", editor.generate_text(prompt))

    print("\n" + "=" * 50)
    print("DEBUG: Let's check what the model predicts for the next token:")

    # Debug: Check the actual logits for the next token
    prompt_ids = editor.tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = editor.model(prompt_ids)
        logits = outputs.logits[0, -1, :]  # Last token's logits

        # Get top predictions
        top_logits, top_indices = torch.topk(logits, 10)
        print("Top 10 predictions:")
        for i, (logit_val, token_id) in enumerate(zip(top_logits, top_indices)):
            token_text = editor.tokenizer.decode([token_id])
            print(f"  {i + 1}. '{token_text}' (ID: {token_id}, logit: {logit_val:.3f})")

        # Specifically check our target tokens
        french_id = editor.tokenizer.encode(" French", add_special_tokens=False)[0]
        english_id = editor.tokenizer.encode(" English", add_special_tokens=False)[0]
        print(f"\nTarget token logits:")
        print(f"  ' French' (ID: {french_id}): {logits[french_id]:.3f}")
        print(f"  ' English' (ID: {english_id}): {logits[english_id]:.3f}")