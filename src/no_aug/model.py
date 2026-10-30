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


class SwigBlock(nn.Module):
    """One residual block:  x + Linear(tanh(Linear(LayerNorm(x))))

    The block expands to `hidden` features, applies tanh, and projects back to
    `width` features so its output can be added to its input (the skip connection).
    """

    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(width)       # pre-norm: normalize before the layers
        self.fc1 = nn.Linear(width, hidden)   # expand:  width  -> hidden
        self.act_proj = nn.Linear(width, hidden)   # expand:  width  -> hidden (activation projection for SwiGLU)
        self.fc2 = nn.Linear(hidden, width)   # project: hidden -> width
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x_norm = self.norm(x)

        # non activation route
        h = self.fc1(x_norm)

        # SwiGLU route
        p = self.act_proj(x_norm)
        gate = torch.nn.functional.silu(p)

        # apply SiLU gate
        activated = h * gate

        activated = self.dropout(activated)

        # bring size back down
        block_out = self.fc2(activated)
        
        return x + block_out                        # skip connection: add the input back
    

class ResidualMLP(nn.Module):
    """Baseline residual MLP.

    input -> Linear(in_features -> width)            project into the backbone
          -> depth x Block                           residual backbone, stays `width` wide
          -> LayerNorm -> Linear -> tanh -> Linear   head: a block without the skip
          -> logits (num_classes)

    The head has no skip connection because its output size (num_classes)
    differs from its input size (width), so the two cannot be added.
    """

    def __init__(self, in_features, num_classes, width=128, hidden=256, depth=40):
        super().__init__()

        # Project the flattened image down to the backbone width.
        self.input_layer = nn.Linear(in_features, width)

        # The residual backbone: `depth` identical blocks.
        self.blocks = nn.ModuleList([SwigBlock(width, hidden) for _ in range(depth)])

        # The head: same layers as a block, but the last layer outputs class scores.
        self.head_norm = nn.LayerNorm(width)
        self.head_fc1 = nn.Linear(width, hidden)
        self.head_fc2 = nn.Linear(hidden, num_classes)

        # Xavier (Glorot) initialization for every linear layer, zero biases.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        lins = [layer for layer in self.modules() if isinstance(layer, nn.Linear)]
        np.savetxt("sample_weights.txt", lins[0].weight.detach().cpu().numpy())

    def forward(self, x, returnHiddenStream=False):
        """Training forward pass. Returns raw logits (no softmax).

        nn.CrossEntropyLoss applies the softmax itself, so training uses logits.
        """
        stream = list()
        x = x.flatten(start_dim=1)            # (N, C, H, W) -> (N, C*H*W)
        x = self.input_layer(x)

        if returnHiddenStream:
            stream.append(x)

        for block in self.blocks:
            x = block(x)

            if returnHiddenStream:
                stream.append(x)

        x = self.head_fc2(torch.nn.functional.silu(self.head_fc1(self.head_norm(x))))

        if returnHiddenStream:
            stream.append(x)

        if returnHiddenStream:
            return x, stream
        return x

    def predict(self, x):
        """Inference pass, never used during training. Returns class probabilities."""
        self.eval()                           # switch layers to inference behavior
        with torch.no_grad():                 # no gradients needed at inference
            return torch.softmax(self.forward(x), dim=1)
