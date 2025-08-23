import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor, CLIPImageProcessor
import os
from . import register_vision_tower
from .base import VisionTower


class CALMProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)


class CALMCrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, anchor_features, augmented_features):
        attn_output, _ = self.cross_attention(
            query=anchor_features,
            key=augmented_features,
            value=augmented_features
        )
        return attn_output


class CALMCompositionModule(nn.Module):
    def __init__(self, anchor_dim, augmented_dim, num_heads=8):
        super().__init__()
        self.projection = CALMProjectionLayer(augmented_dim, anchor_dim)
        self.cross_attention = CALMCrossAttentionLayer(anchor_dim, num_heads)

    def forward(self, anchor_features, augmented_features):
        projected_augmented = self.projection(augmented_features)
        fused_features = self.cross_attention(anchor_features, projected_augmented)
        output = anchor_features + fused_features
        return output


class CALM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Extract model names (assuming they are passed without prefix)
        anchor_model_name = cfg.model_name_or_path
        augmenting_model_name = cfg.model_name_or_path2

        # Load image processors
        self.anchor_image_processor = AutoImageProcessor.from_pretrained(anchor_model_name)
        self.augmenting_image_processor = AutoImageProcessor.from_pretrained(augmenting_model_name)

        # Load both models
        self.anchor_tower = AutoModel.from_pretrained(anchor_model_name)
        self.augmenting_tower = AutoModel.from_pretrained(augmenting_model_name)

        # CALM hyperparameters
        self.num_layers = 4
        self.anchor_fusion_layers = []
        self.augmenting_fusion_layers = []
        self.fusion_layer_map = {}
        self.calm_modules = nn.ModuleList([])

        # Hook related attributes
        self._hooks = []
        self._layer_idx_map = {}  # Maps layer module to its index
        self._augmenting_hidden_states = None  # Stores precomputed augmenting features

        # Update model configs and calculate fusion layers
        self._update_model_configs_and_layers()

    def _fusion_hook(self, module, input, output):
        """
        Hook function to perform CALM fusion at specific layers.
        """
        # 1. Extract the original hidden state from the output tuple
        # Assuming output is a BaseModelOutput or similar, where the first element is hidden_states
        if isinstance(output, tuple):
            original_hidden_state = output[0]
        else:
            # If output is directly the tensor (less common for transformer layers)
            original_hidden_state = output

        # 2. Find the layer index using the pre-built map
        layer_idx = self._layer_idx_map.get(module, None)
        if layer_idx is None:
            # This shouldn't happen if hooks are registered correctly
            print(f"Warning: Hook triggered for unknown module {module}")
            return output

        # 3. Check if this layer is a fusion layer for the anchor model
        if layer_idx in self.anchor_fusion_layers:
            #print(f"activate hook in layer{layer_idx} at [{self.anchor_fusion_layers}]")
            # 4. Find the corresponding augmenting model layer index
            corresponding_augmenting_layer_idx = self.fusion_layer_map[layer_idx]

            # 5. Get the precomputed augmenting features
            # Note: augmenting_tower's layer indexing might be 0-based and continuous,
            # while anchor's might be 1-based if embeddings are considered layer 0.
            # This mapping should be handled in _augmenting_hidden_states indexing.
            # For now, assuming direct index mapping based on layer number.
            # The hidden states list typically includes embeddings as index 0.
            # So, if augmenting_fusion_layers are [3, 6, 9, 12], we fetch from hidden_states[3], etc.
            augmenting_features = self._augmenting_hidden_states[corresponding_augmenting_layer_idx]

            # 6. Get the corresponding CALM module
            fusion_idx = self.anchor_fusion_layers.index(layer_idx)
            calm_module = self.calm_modules[fusion_idx]

            # 7. Perform fusion
            fused_hidden_state = calm_module(original_hidden_state, augmenting_features)

            # 8. Return the fused output in the same format as the original
            # Crucially, we replace the first element (hidden state) with the fused one
            # and keep the rest of the output tuple intact (e.g., attention weights if present)
            if isinstance(output, tuple):
                # Reconstruct the output tuple with the fused hidden state
                new_output = (fused_hidden_state,) + output[1:]
                return new_output
            else:
                # If output was just the tensor, return the fused tensor
                return fused_hidden_state

        # If not a fusion layer, return the original output unchanged
        return output

    def _register_hooks(self):
        """
        Registers forward hooks on the anchor tower's fusion layers.
        This should be called after the model is fully loaded and _update_model_configs_and_layers is done.
        """
        # Clear any existing hooks
        self._clear_hooks()

        # Ensure we have the necessary components
        if self.anchor_tower is None or not hasattr(self.anchor_tower, 'vision_model'):
            raise ValueError("Anchor tower not loaded or does not have a vision_model attribute.")

        # Access the encoder layers of the anchor tower's vision model
        # This is typical for models like CLIPVisionModel
        try:
            layers = self.anchor_tower.vision_model.encoder.layers
        except AttributeError:
            raise ValueError(
                "Could not access encoder layers in anchor_tower.vision_model.encoder.layers. Model structure might be different.")

        # Build the layer index map and register hooks
        for idx, layer in enumerate(layers):
            self._layer_idx_map[layer] = idx + 1  # Assuming layer 0 is embeddings, so transformer layers start from 1

            # Check if this layer is one of our fusion layers
            # Adjust indexing if necessary based on how anchor_fusion_layers was calculated
            # The current logic in _update_model_configs_and_layers sets anchor_fusion_layers starting from 1*gap
            # which aligns with 1-based indexing if we consider embeddings as layer 0.
            # So, if anchor_fusion_layers = [3, 6, 9, 12], we want to hook layers[2], layers[5], etc.
            if (idx + 1) in self.anchor_fusion_layers:
                hook = layer.register_forward_hook(self._fusion_hook)
                self._hooks.append(hook)
                # print(f"Registered hook on anchor tower layer {idx + 1}")

    def _clear_hooks(self):
        """
        Removes all registered hooks.
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._layer_idx_map.clear()

    def _update_model_configs_and_layers(self):
        # Calculate total layers
        self.anchor_total_layers = len(getattr(getattr(self.anchor_tower, 'encoder', None), 'layer', [0] * 12))
        self.augmenting_total_layers = len(getattr(getattr(self.augmenting_tower, 'encoder', None), 'layer', [0] * 12))

        # Calculate fusion layers
        self.anchor_layer_gap = self.anchor_total_layers // self.num_layers
        self.augmenting_layer_gap = self.augmenting_total_layers // self.num_layers

        self.anchor_fusion_layers = [i * self.anchor_layer_gap for i in range(1, self.num_layers + 1)]
        self.augmenting_fusion_layers = [i * self.augmenting_layer_gap for i in range(1, self.num_layers + 1)]
        self.fusion_layer_map = dict(zip(self.anchor_fusion_layers, self.augmenting_fusion_layers))

        # Create CALM modules
        self.calm_modules = nn.ModuleList([
            CALMCompositionModule(
                anchor_dim=self.anchor_tower.config.vision_config.hidden_size,
                augmented_dim=self.augmenting_tower.config.hidden_size
            ) for _ in self.anchor_fusion_layers
        ])

        # Register hooks after updating model configs
        self._register_hooks()

    def forward(self, x, **kwargs):
        # Expecting a dictionary with 'anchor' and 'augmenting' keys
        anchor_input = x['anchor']
        augmenting_input = x['augmenting']

        # Check if models are loaded
        if self.anchor_tower is None or self.augmenting_tower is None:
            raise ValueError("Models not loaded. Call _load_model first.")

        # 1. Pre-compute all augmenting model hidden states
        with torch.no_grad():  # Ensure augmenting_tower doesn't compute gradients
            self.augmenting_tower.eval()  # Set to evaluation mode
            augmenting_outputs = self.augmenting_tower(augmenting_input, output_hidden_states=True)
            # Store the hidden states for use in the hook
            self._augmenting_hidden_states = augmenting_outputs.hidden_states

        # 2. Forward pass through anchor tower
        # The hooks will automatically perform fusion at the specified layers
        anchor_outputs = self.anchor_tower.vision_model(anchor_input)

        # 3. Extract final features from the modified anchor outputs
        # anchor_outputs should now contain the final fused features from the last layer
        final_anchor_features = anchor_outputs.last_hidden_state

        # 4. Clean up temporary stored states
        self._augmenting_hidden_states = None

        # Select features
        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features = final_anchor_features[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features = final_anchor_features
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")

        return image_features


@register_vision_tower('calm')
class CALMVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = CALM(cfg)

        # Set up processors
        self._image_processors = {
            'anchor': self._vision_tower.anchor_image_processor,
            'augmenting': self._vision_tower.augmenting_image_processor
        }
        self._image_processor = self._image_processors

    def _load_model(self, vision_tower_name, **kwargs):
        # Extract model names (assuming they are passed without prefix)
        anchor_model_name = vision_tower_name
        augmenting_model_name = kwargs.get('model_name_or_path2', 'facebook/dinov2-base')

        pretrained_vision_tower_path = kwargs.pop('pretrained_vision_tower_path', None)

        # Freeze the parameters of the base models
        if self._vision_tower.anchor_tower is not None:
            for param in self._vision_tower.anchor_tower.parameters():
                param.requires_grad = False
        if self._vision_tower.augmenting_tower is not None:
            for param in self._vision_tower.augmenting_tower.parameters():
                param.requires_grad = False

        print("Loading CALM vision tower with anchor model from", anchor_model_name, "and augmenting model from",
              augmenting_model_name)
        #(f"vision tower architecture: {self._vision_tower}")
        if pretrained_vision_tower_path is not None:
            # Load from checkpoint
            vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'),
                                              map_location='cpu')
            #print(f"loaded vision tower weights from pretrain {vision_tower_weights}")
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # 1. Extract calm_modules weights
            calm_weights = {k.replace('calm_modules.', ''): v for k, v in vision_tower_weights.items() if
                            k.startswith('calm_modules.')}
            # 2. Load calm_modules weights
            if calm_weights:
                # Load to the calm_modules of the internal CALM instance
                self._vision_tower.calm_modules.load_state_dict(calm_weights)
                print(f"Loaded CALM module weights for {len(calm_weights)} keys from {pretrained_vision_tower_path}.")
            else:
                print(f"Warning: No 'calm_modules' weights found in the checkpoint at {pretrained_vision_tower_path}.")

    def forward(self, x, **kwargs):
        device = x["anchor"].data.device
        self.to(device)
        return self._vision_tower(x, **kwargs)