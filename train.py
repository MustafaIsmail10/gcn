import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_scipy_sparse_matrix
from utils import (
    load_data,
    preprocess_features,
    preprocess_adj,
    chebyshev_polynomials,
)
import scipy.sparse as sp
import matplotlib.pyplot as plt

from model import GCN

# ─── 1. Hyperparameters & Seeds ────────────────────────────────────────────────
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

dataset = "cora"
model_type = "gcn"
learning_rate = 0.01
epochs = 200
hidden1 = 16
dropout = 0.5
weight_decay = 5e-4
early_stopping = 10
max_degree = 3

# ─── 2. Load & Preprocess Data ─────────────────────────────────────────────────
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
    dataset
)

# features: (coords, values, shape)
features = preprocess_features(features)
coords, vals, shape = features
feat_dense = torch.zeros(shape, dtype=torch.float32)
feat_dense[coords[:, 0], coords[:, 1]] = torch.from_numpy(vals).float()

if model_type == "gcn":
    support = [preprocess_adj(adj)]

# We only need the first support to get edge_index
coords, vals, shape = support[0]
# coords is an N×2 array of (row, col) indices, vals is the 1D array of values
adj_sp = sp.coo_matrix((vals, (coords[:, 0], coords[:, 1])), shape=shape)

# Now convert to PyG’s edge_index/edge_weight
edge_index, edge_weight = from_scipy_sparse_matrix(adj_sp)


# Build label tensors

y_all = np.vstack([y_train, y_val, y_test])
y_full = np.zeros_like(y_train)
y_full[train_mask] = y_train[train_mask]
y_full[val_mask] = y_val[val_mask]
y_full[test_mask] = y_test[test_mask]
labels = torch.from_numpy(y_full).float()

# Masks as boolean tensors
train_mask_t = torch.from_numpy(train_mask).bool()
val_mask_t = torch.from_numpy(val_mask).bool()
test_mask_t = torch.from_numpy(test_mask).bool()

# ─── 3. Build Placeholders Dict ────────────────────────────────────────────────
placeholders = {
    "features": feat_dense,
    "labels": labels,
    "labels_mask": train_mask_t,
    "dropout": dropout,
    "weight_decay": weight_decay,
    "support": support,
}

# ─── 4. Instantiate Model & Optimizer ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCN(
    placeholders, input_dim=feat_dense.size(1), hidden_dim=hidden1, logging=True
).to(device)

model._build()
model._built = True

for k, v in placeholders.items():
    if isinstance(v, torch.Tensor):
        placeholders[k] = v.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0,
)


# ─── 5. Evaluation Function ───────────────────────────────────────────────────
def evaluate(mask_t):
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        logits = model(placeholders["features"], edge_index.to(device))
        loss = model.loss(logits, placeholders["labels"], mask_t)
        acc = model.accuracy(logits, placeholders["labels"], mask_t)
    return loss.item(), acc.item(), time.time() - t0


# ─── 6. Training Loop with Early Stopping ──────────────────────────────────────
best_val_loss = float("inf")
wait = 0
val_costs = []

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


print("Starting training...")
for epoch in range(1, epochs + 1):
    t_start = time.time()
    model.train()
    optimizer.zero_grad()

    out = model(placeholders["features"], edge_index.to(device))
    loss = model.loss(out, placeholders["labels"], train_mask_t)
    loss.backward()
    optimizer.step()

    train_acc = model.accuracy(out, placeholders["labels"], train_mask_t).item()
    train_losses.append(loss.item())
    train_accuracies.append(train_acc)

    val_loss, val_acc, val_time = evaluate(val_mask_t)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch:04d} | "
        f"Train Loss: {loss.item():.5f} | Train Acc: {train_acc:.5f} | "
        f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.5f} | "
        f"Time: {time.time() - t_start:.4f}s"
    )

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        wait += 1
        if wait > early_stopping:
            print("Early stopping...")
            break

print("Training completed. Loading best model...")
model.load_state_dict(torch.load("best_model.pt"))

# ─── 7. Final Test Evaluation ─────────────────────────────────────────────────
test_loss, test_acc, test_time = evaluate(test_mask_t)
print(f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f} | Time: {test_time:.4f}s")

# ─── 8. Plot Training & Validation Metrics ─────────────────────────────────────
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy")
plt.axhline(y=test_acc, color="r", linestyle="--", label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
