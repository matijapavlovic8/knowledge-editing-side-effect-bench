import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

class GPT2ROMEEditor:
    def __init__(self, model_name="gpt2-xl", device="cuda"):
        self.device = device
        print(f"Loading model {model_name} on {device}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print("Model loaded.")

    def _get_activations_and_jacobian(self, input_ids, layer_idx):
        """
        Get activations and Jacobian of the MLP c_fc layer output w.r.t. c_fc weights.
        We compute: output = c_fc(input) = W x + b
        Jacobian of output w.r.t. W is the input x.

        We'll:
        - register hook to get input to c_fc layer (for Jacobian)
        - forward pass to get output activations
        """

        activations = {}
        mlp_inputs = {}

        def forward_hook(module, input, output):
            # input is a tuple, input[0] is the input tensor to c_fc layer
            mlp_inputs["input"] = input[0].detach()

        def output_hook(module, input, output):
            activations["output"] = output.detach()

        c_fc_layer = self.model.transformer.h[layer_idx].mlp.c_fc

        handle_in = c_fc_layer.register_forward_hook(forward_hook)
        handle_out = c_fc_layer.register_forward_hook(output_hook)

        with torch.no_grad():
            _ = self.model(input_ids)

        handle_in.remove()
        handle_out.remove()

        if "input" not in mlp_inputs or "output" not in activations:
            raise RuntimeError("Failed to capture activations or inputs for jacobian.")

        return activations["output"], mlp_inputs["input"]

    def _compute_jacobian_update(self, old_act, new_act, mlp_input):
        """
        Compute rank-1 update on c_fc weights using Jacobian method:

        Given:
        - old_act: output activations before edit (shape [seq_len, hidden_size * 4])
        - new_act: output activations after edit (same shape)
        - mlp_input: input to c_fc layer (shape [seq_len, hidden_size])

        The weight update ΔW solves ΔW @ mlp_input.T = new_act - old_act

        We approximate ΔW as a rank-1 update: u v^T

        Solve for u and v:

        - v = average mlp_input vector (shape hidden_size)
        - u = (mean delta activations) / (norm of v)^2

        So that ΔW ≈ u @ v^T minimizes squared error.

        """

        # Flatten batch and sequence dims (assuming batch=1 here)
        delta_act = (new_act - old_act).mean(dim=0)  # (4*hidden_size,)
        v = mlp_input.mean(dim=0)  # (hidden_size,)

        v_norm_sq = (v @ v).item()
        if v_norm_sq < 1e-10:
            raise RuntimeError("Norm of input vector is too small for stable update.")

        u = delta_act / v_norm_sq  # (4*hidden_size,)

        # Reshape to column and row vectors
        u = u.unsqueeze(1)  # (4*hidden_size, 1)
        v = v.unsqueeze(0)  # (1, hidden_size)

        return u, v

    def _apply_rank1_update(self, layer_idx, u, v):
        """
        Apply the rank-1 update u v^T to the c_fc weights of the chosen layer.
        """
        with torch.no_grad():
            W = self.model.transformer.h[layer_idx].mlp.c_fc.weight.data
            print(f"Layer {layer_idx} c_fc weight norm before update: {W.norm().item():.4f}")
            W += u @ v
            print(f"Layer {layer_idx} c_fc weight norm after update: {W.norm().item():.4f}")

    def _compute_edit_impact(self, old_act, new_act):
        """
        Compute scalar impact of the edit at one layer by averaging norm of activation difference.
        """
        return torch.norm(new_act - old_act, dim=-1).mean().item()

    def find_best_layer(self, subject, relation, old_object, new_object, candidate_layers=None):
        """
        For each candidate layer, compute the activation difference magnitude
        between old and new prompts and select the layer with maximum difference.

        Returns: best_layer_idx, dict of layer->impact
        """
        if candidate_layers is None:
            candidate_layers = list(range(20, 36))  # GPT2 XL layers with MLPs

        old_prompt = f"{subject} {relation} {old_object}"
        new_prompt = f"{subject} {relation} {new_object}"

        old_ids = self.tokenizer(old_prompt, return_tensors="pt").input_ids.to(self.device)
        new_ids = self.tokenizer(new_prompt, return_tensors="pt").input_ids.to(self.device)

        layer_impacts = {}
        for layer in candidate_layers:
            old_act, _ = self._get_activations_and_jacobian(old_ids, layer)
            new_act, _ = self._get_activations_and_jacobian(new_ids, layer)
            impact = self._compute_edit_impact(old_act, new_act)
            layer_impacts[layer] = impact

        best_layer = max(layer_impacts, key=layer_impacts.get)
        print(f"Best layer for edit impact: {best_layer} with impact {layer_impacts[best_layer]:.4f}")
        return best_layer, layer_impacts

    def edit_fact(self, subject, relation, old_object, new_object, layer_idx=None):
        """
        Edit a single fact by performing Jacobian rank-1 update on the best layer if
        layer_idx not provided.
        """
        if layer_idx is None:
            layer_idx, _ = self.find_best_layer(subject, relation, old_object, new_object)

        old_prompt = f"{subject} {relation} {old_object}"
        new_prompt = f"{subject} {relation} {new_object}"

        old_ids = self.tokenizer(old_prompt, return_tensors="pt").input_ids.to(self.device)
        new_ids = self.tokenizer(new_prompt, return_tensors="pt").input_ids.to(self.device)

        old_act, mlp_input_old = self._get_activations_and_jacobian(old_ids, layer_idx)
        new_act, mlp_input_new = self._get_activations_and_jacobian(new_ids, layer_idx)

        # Use average mlp_input for update stability
        mlp_input = (mlp_input_old + mlp_input_new) / 2

        u, v = self._compute_jacobian_update(old_act, new_act, mlp_input)

        self._apply_rank1_update(layer_idx, u, v)

    def batch_edit(self, edits, candidate_layers=None):
        """
        Batch edit multiple facts.

        edits: list of dicts, each with keys: subject, relation, old_object, new_object

        For each edit:
        - find best layer (or use candidate_layers to restrict search)
        - compute u,v updates
        - accumulate u,v per layer as sum (or average)
        - apply updates per layer after all are computed
        """
        if candidate_layers is None:
            candidate_layers = list(range(20, 36))

        # Initialize dict to accumulate updates by layer
        updates = {layer: {"u": None, "v": None, "count": 0} for layer in candidate_layers}

        for edit in edits:
            subject = edit["subject"]
            relation = edit["relation"]
            old_obj = edit["old_object"]
            new_obj = edit["new_object"]

            best_layer, _ = self.find_best_layer(subject, relation, old_obj, new_obj, candidate_layers)

            old_prompt = f"{subject} {relation} {old_obj}"
            new_prompt = f"{subject} {relation} {new_obj}"

            old_ids = self.tokenizer(old_prompt, return_tensors="pt").input_ids.to(self.device)
            new_ids = self.tokenizer(new_prompt, return_tensors="pt").input_ids.to(self.device)

            old_act, mlp_input_old = self._get_activations_and_jacobian(old_ids, best_layer)
            new_act, mlp_input_new = self._get_activations_and_jacobian(new_ids, best_layer)

            mlp_input = (mlp_input_old + mlp_input_new) / 2

            u, v = self._compute_jacobian_update(old_act, new_act, mlp_input)

            # Accumulate updates: simple sum of u and v scaled by count
            if updates[best_layer]["u"] is None:
                updates[best_layer]["u"] = u
                updates[best_layer]["v"] = v
            else:
                updates[best_layer]["u"] += u
                updates[best_layer]["v"] += v

            updates[best_layer]["count"] += 1

        # Apply average updates per layer
        for layer, data in updates.items():
            if data["count"] > 0:
                u_avg = data["u"] / data["count"]
                v_avg = data["v"] / data["count"]
                print(f"Applying batch update to layer {layer}, averaged over {data['count']} edits.")
                self._apply_rank1_update(layer, u_avg, v_avg)

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        print(f"Saving model to {path}...")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print("Model saved.")

    def load_model(self, path):
        print(f"Loading model from {path}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.model = GPT2LMHeadModel.from_pretrained(path).to(self.device)
        self.model.eval()
        print("Model loaded.")

    def generate_text(self, prompt, max_length=50):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    editor = GPT2ROMEEditor(device=device)

    prompt = "Danielle Darrieux's mother tongue is"
    print("Before edit:", editor.generate_text(prompt))

    # Single edit
    editor.edit_fact("Danielle Darrieux", "'s mother tongue is", "French", "English")

    print("After single edit:", editor.generate_text(prompt))

    # Batch edit example
    batch_edits = [
        {
            "subject": "Danielle Darrieux",
            "relation": "'s mother tongue is",
            "old_object": "French",
            "new_object": "English",
        },
        {
            "subject": "Edwin of Northumbria",
            "relation": "'s religion is",
            "old_object": "Christianity",
            "new_object": "Islam",
        },
    ]
    editor.batch_edit(batch_edits)
    print("After batch edit:", editor.generate_text(prompt))

    editor.save_model("./edited_gpt2xl")
