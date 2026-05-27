"""
Klasifikasi digit dengan PyTorch — pola training standar yang akan kamu pakai
berulang kali. Memakai dataset 'digits' dari scikit-learn (8x8, ringan, tanpa
unduhan) agar bisa langsung jalan di CPU.

Menunjukkan: Dataset/DataLoader, nn.Module, training loop, validasi,
            train()/eval(), no_grad().

Jalankan:  python pytorch_mnist.py
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ---------- 1. Data ----------
digits = load_digits()
X, y = digits.data, digits.target            # X: (1797, 64), y: 0..9

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

# Ubah ke tensor & bungkus dalam DataLoader (untuk mini-batch + shuffle)
train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                         torch.tensor(y_tr, dtype=torch.long))
test_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                        torch.tensor(y_te, dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)


# ---------- 2. Model ----------
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),                 # 10 kelas; CrossEntropyLoss pakai logit
        )

    def forward(self, x):
        return self.net(x)


model = DigitClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- 3. Helper evaluasi ----------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return correct / total


# ---------- 4. Training loop ----------
EPOCHS = 30
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()          # reset gradien
        loss = criterion(model(xb), yb)
        loss.backward()                # backprop
        optimizer.step()               # update bobot
        running_loss += loss.item()

    if epoch % 5 == 0:
        acc = evaluate(test_loader)
        print(f"Epoch {epoch:2d} | loss={running_loss/len(train_loader):.4f} "
              f"| test_acc={acc:.4f}")

print(f"\nAkurasi test akhir: {evaluate(test_loader):.4f}")
