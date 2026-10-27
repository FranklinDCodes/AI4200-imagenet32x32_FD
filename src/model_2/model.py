import torch
import numpy as np
import torch.nn as nn
from math import sqrt

# Each task differs only in its input size and its number of classes.
# The same model class is used for all three.
TASKS = {
    "mnist":      {"in_features": 28 * 28,     "num_classes": 10},    # 28x28 grayscale
    "cifar10":    {"in_features": 3 * 32 * 32, "num_classes": 10},    # 32x32 color
    "imagenet32": {"in_features": 3 * 32 * 32, "num_classes": 1000},  # 32x32 color
}


class Block(nn.Module):
    """One residual block:  x + Linear(tanh(Linear(LayerNorm(x))))

    The block expands to `hidden` features, applies tanh, and projects back to
    `width` features so its output can be added to its input (the skip connection).
    """

    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(width)       # pre-norm: normalize before the layers
        self.fc1 = nn.Linear(width, hidden)   # expand:  width  -> hidden
        self.fc2 = nn.Linear(hidden, width)   # project: hidden -> width

    def forward(self, x):
        h = self.fc2(torch.tanh(self.fc1(self.norm(x))))
        return x + h                          # skip connection: add the input back


class ResidualMLP(nn.Module):
    """Baseline residual MLP.

    input -> Linear(in_features -> width)            project into the backbone
          -> depth x Block                           residual backbone, stays `width` wide
          -> LayerNorm -> Linear -> tanh -> Linear   head: a block without the skip
          -> logits (num_classes)

    The head has no skip connection because its output size (num_classes)
    differs from its input size (width), so the two cannot be added.
    """

    def __init__(self, in_features, num_classes, width=128, hidden=256, depth=30):
        super().__init__()

        # Project the flattened image down to the backbone width.
        self.input_layer = nn.Linear(in_features, width)

        # The residual backbone: `depth` identical blocks.
        self.blocks = nn.ModuleList([Block(width, hidden) for _ in range(depth)])

        # The head: same layers as a block, but the last layer outputs class scores.
        self.head_norm = nn.LayerNorm(width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)

        # Xavier (Glorot) initialization for every linear layer, zero biases.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                self._init_linear(layer)

        lins = [layer for layer in self.modules() if isinstance(layer, nn.Linear)]
        np.savetxt("sample_weights.txt", lins.detach().cpu().numpy())

    # initializes all the weight layers
    def _init_linear(self, layer: nn.Linear):

        fan_in, fan_out = layer.in_features, layer.out_features
        W = self._InterNVG(fan_in, fan_out)

        with torch.no_grad():
            layer.weight.copy_(W.T) # nn.Linear stores (out, in), so transpose
            if layer.bias is not None:
                layer.bias.zero_()

    # generates a matrix with increased Inter-neuron variance with Glorot (InterNVG)
    def _InterNVG(
            self,
            fan_in: int, 
       	    fan_out: int,
       	    starting_feature_var: float = 2.0,
            starting_neuron_var: float = 0.1
        ) -> torch.Tensor:

        # create sample row, shape (FAN_OUT,)
        sample_feat_vec = torch.randn(fan_out) * sqrt(starting_feature_var)

        # generate the full weight matrix in one shot, shape (FAN_IN, FAN_OUT)
        # each column j is noise + sample_feat_vec[j] (broadcast across rows)
        weight_mat_raw = torch.randn(fan_in, fan_out) * starting_neuron_var + sample_feat_vec
        glorot_var = 2 / (fan_in + fan_out)

        # bring back to desired distribution
        weight_mat = torch.sqrt(
        glorot_var / torch.var(weight_mat_raw, unbiased=False)
        ) * (weight_mat_raw - weight_mat_raw.mean())

        return weight_mat

    def forward(self, x):
        """Training forward pass. Returns raw logits (no softmax).

        nn.CrossEntropyLoss applies the softmax itself, so training uses logits.
        """
        x = x.flatten(start_dim=1)            # (N, C, H, W) -> (N, C*H*W)
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_fc2(torch.tanh(self.head_fc1(self.head_norm(x))))
        return x

    def predict(self, x):
        """Inference pass, never used during training. Returns class probabilities."""
        self.eval()                           # switch layers to inference behavior
        with torch.no_grad():                 # no gradients needed at inference
            return torch.softmax(self.forward(x), dim=1)
