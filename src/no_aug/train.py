import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import TASKS, ResidualMLP


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
LEARNING_RATE = 0.003
BATCH_SIZE = 2048
EPOCHS = {"mnist": 30, "cifar10": 10, "imagenet32": 20}



def prepare(X):
    """Convert uint8 images (0-255) to floats in roughly [-1, 1]."""

    X_normed = (torch.as_tensor(X).float() / 255.0 - 0.5) / 0.5

    return X_normed


# ---------------------------------------------------------------------------
# The two functions below are the REQUIRED INTERFACE for every submission.
# The grading notebook calls them like this:
#
#     model = train("cifar10", X_train, y_train, seed=0, device="cuda")
#     probs = predict(model, X_test)
#
# Change anything inside them, but keep their names and arguments.
# ---------------------------------------------------------------------------

def get_activation_variance(model, input):

    yhat, stream = model(input, True)

    vars = list()
    for h in stream:

        vars.append(h.std(axis=1).mean().item())

    return vars

def train(task, X_train, y_train, seed=0, device="cuda"):
    """Build a new model and train it from scratch. Returns the trained model."""
    torch.manual_seed(seed)                   # makes the run reproducible

    # 1. Build the model for this task. Only input size and class count differ.
    model = ResidualMLP(**TASKS[task]).to(device)

    eval_set = X_train[:1024]
    X_train = X_train[1024:]
    y_train = y_train[1024:]

    # 2. Wrap the data so we can loop over shuffled mini-batches.
    dataset = TensorDataset(prepare(X_train), torch.as_tensor(y_train).long())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # create weights based on downscaled class count z scores
    t_counts = torch.unique(torch.as_tensor(y_train).long(), return_counts=True)[1]
    t_class_weights = 1 - (t_counts - t_counts.float().mean()) / t_counts.float().mean()
    print(f"Class counts: {t_counts.detach().cpu().numpy().tolist()}")
    print(f"Class weights: {t_class_weights.detach().cpu().numpy().tolist()}")

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss(weight=t_class_weights.to(device))           # expects raw logits # weight=t_class_weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # d_vars = dict()
    # d_vars["start"] = get_activation_variance(model, eval_set)

    # 4. The training loop.
    model.train()
    for epoch in range(EPOCHS[task]):


        # grab variance
        # if start or midpoint
        # if epoch == 0 or epoch == (EPOCHS[task] + 1) // 2:
        #     d_vars["mid"] = get_activation_variance(model, eval_set)

        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)           # forward pass
            loss = loss_fn(logits, y_batch)   # how wrong are we?

            optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients
            optimizer.step()                  # update the weights

            total_loss += loss.item() * len(X_batch)

        print(f"epoch {epoch + 1}: train loss {total_loss / len(dataset):.4f}")

    # d_vars["end"] = get_activation_variance(model, eval_set)

    return model


def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)

