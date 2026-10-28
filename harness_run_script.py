# ---- Your submission: change these four lines -----------------------------

# Your GitHub repository, in the form pip installs from. This example is the
# instructor's baseline: https://github.com/JesseTNRoberts/AI4200-imagenet32x32-baseline-model
#PACKAGE_URL = "git+https://github.com/FranklinDCodes/AI4200-imagenet32x32_FD.git"
PACKAGE_NAME = "model_shift"

# The name you import your package by (the folder inside src/).
IMPORT_NAME = "model_2"

# Your nickname on the unofficial leaderboard (not your real name).
# Set it to "" to keep your results off the leaderboard.
LEADERBOARD_ALIAS = "Frankie_secret_init"

# Your section: "4000" or "5000".
SECTION = "4000"

import os

# Where the datasets are saved. Keep this folder between runs to skip downloading.
DATA_DIR = "data"

# Which tasks to run. Remove some while testing to save time.
TASKS_TO_RUN = ["mnist"] #, "cifar10", "imagenet32"]

SEED = 0

# Every finished task adds one row to this file.
RESULTS_FILE = "results.csv"

# The unofficial leaderboard's Google Form.
LEADERBOARD_FORM = "https://docs.google.com/forms/d/e/1FAIpQLScZjEdbLFtn1CNvsfwYEbM4YGRxdOU5cyJ7RANM7MrxT0a4oA/viewform?usp=dialog"
BASELINE_URL = "git+https://github.com/JesseTNRoberts/AI4200-imagenet32x32-baseline-model.git"

# The instructor's grading script runs this notebook for every student and
# sets these through environment variables instead of editing the cells.
#PACKAGE_URL = os.environ.get("HARNESS_PACKAGE_URL", PACKAGE_URL)
IMPORT_NAME = os.environ.get("HARNESS_IMPORT_NAME", IMPORT_NAME)
LEADERBOARD_ALIAS = os.environ.get("HARNESS_LEADERBOARD_ALIAS", LEADERBOARD_ALIAS)
DATA_DIR = os.environ.get("HARNESS_DATA_DIR", DATA_DIR)
RESULTS_FILE = os.environ.get("HARNESS_RESULTS_FILE", RESULTS_FILE)
TASKS_TO_RUN = os.environ.get("HARNESS_TASKS", ",".join(TASKS_TO_RUN)).split(",")

import importlib
import time
import zipfile

import numpy as np
import torch

student = importlib.import_module(f"src.{PACKAGE_NAME}")   # your package
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Testing", IMPORT_NAME, "on", DEVICE)

# For each task: Hugging Face dataset name, image column, train split, test split.
SOURCES = {
    "mnist":      ("ylecun/mnist",                     "image", "train", "test"),
    "cifar10":    ("uoft-cs/cifar10",                  "img",   "train", "test"),
    "imagenet32": ("benjamin-paine/imagenet-1k-32x32", "image", "train", "validation"),
}
NUM_CLASSES = {"mnist": 10, "cifar10": 10, "imagenet32": 1000}


def save_split(split, image_column, mode, folder, name):
    """Write one Hugging Face split to X_<name>.npy (N, C, H, W) and y_<name>.npy.

    The images are written straight into a file on disk, 10,000 at a time, so
    even ImageNet (about 4 GB) never has to fit in memory all at once.
    """
    n = len(split)
    height, width = np.asarray(split[0][image_column]).shape[:2]
    channels = 1 if mode == "L" else 3
    X = np.lib.format.open_memmap(os.path.join(folder, f"X_{name}.npy"), mode="w+",
                                  dtype=np.uint8, shape=(n, channels, height, width))
    y = np.empty(n, dtype=np.int64)
    for start in range(0, n, 10_000):
        chunk = split[start:start + 10_000]
        # mode is "L" (grayscale) or "RGB"; convert() fixes any odd image formats.
        images = np.stack([np.asarray(img.convert(mode)) for img in chunk[image_column]])
        stop = start + len(images)
        X[start:stop] = images[:, None] if mode == "L" else images.transpose(0, 3, 1, 2)
        y[start:stop] = chunk["label"]
        print(f"  {name}: {stop:,} / {n:,} images", end="\r")
    print()
    X.flush()
    np.save(os.path.join(folder, f"y_{name}.npy"), y)


def load_task(task):
    """Return X_train, y_train, X_test, y_test, downloading only if there is no saved copy.

    The arrays are memory-mapped: they are read from disk as they are used
    instead of all being loaded into RAM up front.
    """
    folder = os.path.join(DATA_DIR, task)
    old_file = os.path.join(DATA_DIR, f"{task}.npz")   # saved by an earlier version of this notebook

    if not os.path.isdir(folder) and os.path.exists(old_file):
        # Unpack the earlier single-file copy into the folder format; no download needed.
        print(f"{task}: unpacking {old_file} (one time only; you can delete it afterwards)")
        temp_folder = folder + ".partial"
        os.makedirs(temp_folder, exist_ok=True)
        # An .npz file is a zip of .npy files, so extracting it streams them to disk
        # without loading the arrays into memory.
        with zipfile.ZipFile(old_file) as old:
            old.extractall(temp_folder)
        os.replace(temp_folder, folder)

    if not os.path.isdir(folder):                    # no saved copy yet: download it
        from datasets import load_dataset
        name, image_column, train_split, test_split = SOURCES[task]
        mode = "L" if task == "mnist" else "RGB"
        print(f"{task}: getting {name} from Hugging Face (first run only; reuses its download cache if present)")

        # Write into a temporary folder, then rename it. An interrupted run then
        # never leaves a half-written folder that looks like a finished one.
        temp_folder = folder + ".partial"
        os.makedirs(temp_folder, exist_ok=True)
        save_split(load_dataset(name, split=train_split), image_column, mode, temp_folder, "train")
        save_split(load_dataset(name, split=test_split), image_column, mode, temp_folder, "test")
        os.replace(temp_folder, folder)

    # mmap_mode="c" (copy-on-write): read from disk on demand, and any changes
    # your code makes to the arrays stay in memory and never touch the files.
    return tuple(np.load(os.path.join(folder, f"{part}.npy"), mmap_mode="c")
                 for part in ["X_train", "y_train", "X_test", "y_test"])


# Download (or load) every task once and check what we got.
for task in TASKS_TO_RUN:
    X_train, y_train, X_test, y_test = load_task(task)
    print(f"{task:>10}: X_train {X_train.shape} {X_train.dtype}, X_test {X_test.shape}, "
          f"labels {y_train.min()}..{y_train.max()}")
    del X_train, y_train, X_test, y_test             # free memory before the next one# A quick look at a few training images from each task.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(TASKS_TO_RUN), 8, figsize=(10, 1.4 * len(TASKS_TO_RUN)), squeeze=False)
for row, task in enumerate(TASKS_TO_RUN):
    X_train, y_train, _, _ = load_task(task)
    for col in range(8):
        img = X_train[col].transpose(1, 2, 0)          # (C, H, W) -> (H, W, C) for plotting
        axes[row, col].imshow(img.squeeze(), cmap="gray" if img.shape[2] == 1 else None)
        axes[row, col].set_title(f"{task}\nlabel {y_train[col]}" if col == 0 else f"label {y_train[col]}", fontsize=8)
        axes[row, col].axis("off")
    del X_train, y_train
plt.tight_layout()
plt.show()


import json
import re
import urllib.parse
import urllib.request
import csv

LEADERBOARD_QUESTIONS = ["alias", "section", "task", "accuracy", "minute", "repo"]
_form_fields = None   # the form's field names, looked up once


def find_form_fields():
    """Return the form's field names (entry.NNN), one per question, in the order above."""
    address, _, query = LEADERBOARD_FORM.partition("?")
    entries = [key for key, _ in urllib.parse.parse_qsl(query) if key.startswith("entry.")]
    if len(entries) >= len(LEADERBOARD_QUESTIONS):           # a pre-filled link lists them
        return entries[:len(LEADERBOARD_QUESTIONS)]
    # Otherwise read them from the form page itself.
    page = urllib.request.urlopen(address, timeout=20).read().decode("utf-8")
    data = json.loads(re.search(r"FB_PUBLIC_LOAD_DATA_ = (.*?);\s*</script>", page, re.S).group(1))
    questions = [(q[1] or "", f"entry.{q[4][0][0]}") for q in data[1][1] if q[4]]
    fields = []
    for position, keyword in enumerate(LEADERBOARD_QUESTIONS):   # match by title, else by position
        match = [entry for title, entry in questions if keyword in title.lower()]
        fields.append(match[0] if match else questions[position][1])
    return fields


def post_to_leaderboard(task, accuracy, train_minutes):
    """Send one result to the unofficial leaderboard. A failure only prints a warning."""
    global _form_fields
    alias = LEADERBOARD_ALIAS.strip()
    if not alias or not LEADERBOARD_FORM:
        return
    if alias.lower() == "baseline" and PACKAGE_URL != BASELINE_URL:
        print("  not posted: set LEADERBOARD_ALIAS to your own nickname to join the leaderboard")
        return
    answers = [alias, SECTION, task, f"{accuracy:.4f}", f"{train_minutes:.1f}", PACKAGE_URL]
    try:
        if _form_fields is None:
            _form_fields = find_form_fields()
        address = LEADERBOARD_FORM.partition("?")[0].replace("/viewform", "/formResponse")
        data = urllib.parse.urlencode(dict(zip(_form_fields, answers))).encode()
        urllib.request.urlopen(address, data=data, timeout=20)
        print(f"  posted to the leaderboard as {alias}")
    except Exception as error:
        print(f"  could not post to the leaderboard ({error}); the result is still in {RESULTS_FILE}")

results = []

for task in TASKS_TO_RUN:
    print(f"\n===== {task} =====")
    X_train, y_train, X_test, y_test = load_task(task)

    # Train from scratch.
    start = time.time()
    model = student.train(task, X_train, y_train, seed=SEED, device=DEVICE)
    train_minutes = (time.time() - start) / 60

    # Predict on the held-out test set.
    probs = torch.as_tensor(student.predict(model, X_test)).float().cpu()

    # Check that predict() returned one probability row per test image.
    assert probs.shape == (len(y_test), NUM_CLASSES[task]), f"predict() returned shape {tuple(probs.shape)}"
    assert torch.allclose(probs.sum(dim=1), torch.ones(len(probs)), atol=1e-3), "rows of predict() must sum to 1"

    accuracy = (probs.argmax(dim=1).numpy() == y_test).mean()
    results.append((task, accuracy, train_minutes))
    print(f"{task}: test accuracy {accuracy:.2%}  (training took {train_minutes:.1f} min)")

    # Save the result to a file right away, in case the browser stops showing output.
    new_file = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "package_url", "import_name", "task", "accuracy", "train_minutes", "seed"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M"), PACKAGE_NAME, IMPORT_NAME, task,
                         f"{accuracy:.4f}", f"{train_minutes:.1f}", SEED])
    # post_to_leaderboard(task, accuracy, train_minutes)

    del model, X_train, y_train, X_test, y_test, probs   # free memory before the next task
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print(f"{'task':>10} | {'test accuracy':>13} | {'train time':>10}")
print("-" * 41)
for task, accuracy, minutes in results:
    print(f"{task:>10} | {accuracy:>13.2%} | {minutes:>6.1f} min")
