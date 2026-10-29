
### #1 - Claude Sonnet 5.5 Extra - shifting augmentation

**Me:**
Hey Claude. Can you write a python torch function that takes in a tensor of shape (B, C, H, W) and creates k new copies of the dataset where each H, W is either row or column shifted with wrap around by a random value between 0-x_max_shift or 0-y_max_shit (values for rows or columns).  I want this function to ensure that each shifted version is unique (sample shifts for each k without replacement. I want it to return the original dataset concatenated with all the new instances and a label tensor that is based on an input label tensor but propagated for all the new distorted images. Thanks!

**Claude:**
I'll write this as a file and run a quick test on it.

I wrote `shift_augment.py` with the function, but I haven't run it. The sandbox has no PyTorch and no network access to install it. The file ends with a self-test block (`python shift_augment.py`) that checks shapes, ordering, label propagation, uniqueness and the error case.

Here is how I read your spec, in case you meant something different:

- One shift per copy. Copy `j` gets a single `(dy, dx)` pair applied to the whole dataset. `dy` is a row shift in `[0, y_max_shift]` and `dx` is a column shift in `[0, x_max_shift]`. Rolling is done with `torch.roll` over dims `(2, 3)`.
- Unique shifts. The `k` pairs are drawn without replacement from the grid of all `(dy, dx)` combinations, using `randperm` over a flat index.
- No identity shift by default. `(0, 0)` is excluded because it would duplicate the original images. Pass `allow_identity=True` if you want it as a possible draw.
- Shift caps. Shifts are capped at `H-1` and `W-1`, because under wrap-around a shift of `H` equals a shift of 0. Without the cap, two "different" shifts could give identical images.
- Too many copies. If `k` is more than the number of unique shifts available, it raises a `ValueError` instead of silently repeating.

The output is `(x_out, labels_out)` with shapes `(B*(k+1), C, H, W)` and `(B*(k+1), ...)`. The order is the original dataset first, then copy 1 through copy `k`, with labels repeated in the same order. Labels can be any shape as long as the first dimension is `B`, so class indices and soft labels both work. For reproducibility, pass a CPU `torch.Generator`.

If you meant each image in the batch to get its own independent random shift, instead of one shift per copy, that is a different design, and I can rework it that way.

**Me:**
Yes, could you please make it so that each image has its own unique shift per each k copy of the dataset. Also, I forgot to mention that I would like negative shifts too. So really (-x_max_shift, x_max_shift) is the appropriate interval to sample from for x, e.g.

**Claude:**
Torch still isn't available here, so I'll check the core logic (the gather indexing, the identity-skipping, and the wrap-collapse) with a NumPy mirror.

I've rewritten `shift_augment.py` with both changes. I still can't run PyTorch here, so the function itself and its built-in self-test (`python shift_augment.py`) haven't been run. I did check the core logic with a NumPy mirror: the per-image gather matches `np.roll`, the identity-skipping works, and the wrap-around collapsing holds. Run the self-test on your end to confirm.

What changed

- Per-image shifts. In each of the `k` copies, every image gets its own `(dy, dx)`. For a given image, its `k` shifts are sampled without replacement, so its `k` versions are all distinct. Different images are sampled independently.
- Signed shifts. `dy` is drawn from `[-y_max_shift, y_max_shift]` and `dx` from `[-x_max_shift, x_max_shift]`. Positive `dy` moves content down and positive `dx` moves it right, as with `torch.roll`. Negative values move it up or left. The number of available combinations is `(2*y_max+1)*(2*x_max+1) - 1`, since `(0, 0)` is still excluded by default (`allow_identity=True` lets it be drawn).
- Wrap-around duplicates. If `2*max_shift + 1` is larger than `W` (or `H`), some shifts become equivalent, for example +1 and −(W−1). Those are collapsed so uniqueness still holds, which means at most `W` distinct column shifts and `H` distinct row shifts. If `k` is more than the number of distinct shifts available, it raises a `ValueError`.

Per-image shifts can't be done with one `torch.roll` per copy, so each copy is built with a single gather (index arithmetic with `% H` and `% W`) across all `B` images. That keeps the loop at `k` iterations instead of `B*k` rolls. Sampling uses top-k over random keys in chunks, so memory stays bounded even with large shift ranges.

The output order is unchanged: the original dataset first, then copy 1 through copy `k`, with labels repeated to match. The function doesn't return the shifts it used. If you want them, for logging or for using them as extra targets, say so and I'll add an option for it.

**Resulting code was used**



### #2 - Claude Sonnet 5.5 Extra - flipping augmentation

**Me:**
Hey Claude. Can you write a python torch function that takes in a tensor of shape (B, C, H, W) and creates 2 new versions of each instance: one flipped vertically, and one flipped horizontally. By flipped I mean visually (e.g. for vertical, this means reversing all the rows). I want it to return the original dataset concatenated with all the new instances and a label tensor that is based on an input label tensor but propagated for all the new flipped images as well. Thanks!

**Claude:**
```python
import torch

def augment_with_flips(images: torch.Tensor, labels: torch.Tensor):
    """
    images: (B, C, H, W)
    labels: (B, ...) e.g. (B,) for class labels
    returns: images of shape (3B, C, H, W), labels of shape (3B, ...)
    """
    v_flipped = torch.flip(images, dims=[2])  # reverse rows (H axis)
    h_flipped = torch.flip(images, dims=[3])  # reverse columns (W axis)

    all_images = torch.cat([images, v_flipped, h_flipped], dim=0)
    all_labels = torch.cat([labels, labels, labels], dim=0)
    return all_images, all_labels
```

The output is ordered in three blocks: the original `B` images first, then the `B` vertically flipped ones, then the `B` horizontally flipped ones. The label tensor uses the same layout, so `all_labels[i]` always matches `all_images[i]`.

Since `torch.cat` works along dim 0, the labels can have any trailing shape, such as `(B,)` class indices or `(B, num_classes)` one-hot vectors.

This assumes the label doesn't change under a flip, as with classification. If your labels are spatial, like segmentation masks or bounding boxes, they'd need to be flipped the same way as the images.

**This code provided was used**
