"""
Neural Network dari NOL — hanya NumPy.

MLP 2-layer untuk klasifikasi biner. Mengimplementasikan forward pass,
backpropagation (chain rule manual), dan training loop. Tujuannya membuka
"black box" deep learning.

Arsitektur:  input(2) -> hidden(8, ReLU) -> output(1, sigmoid)
Dataset    :  "make_moons" — dua kelas berbentuk bulan sabit (non-linear).

Jalankan:  python neural_network_scratch.py
"""
import numpy as np


# ---------- Fungsi aktivasi & turunannya ----------
def relu(z):        return np.maximum(0, z)
def relu_grad(z):   return (z > 0).astype(float)
def sigmoid(z):     return 1 / (1 + np.exp(-z))


class NeuralNetwork:
    def __init__(self, n_input, n_hidden, lr=0.1, seed=42):
        rng = np.random.default_rng(seed)
        # Inisialisasi He untuk layer ReLU (skala sesuai fan-in)
        self.W1 = rng.normal(0, np.sqrt(2 / n_input), (n_input, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, 1))
        self.b2 = np.zeros(1)
        self.lr = lr

    def forward(self, X):
        # Simpan nilai antara untuk dipakai saat backprop
        self.X = X
        self.z1 = X @ self.W1 + self.b1      # pra-aktivasi layer 1
        self.a1 = relu(self.z1)              # aktivasi layer 1
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)           # output (probabilitas)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        # Binary cross-entropy
        eps = 1e-9
        y_true = y_true.reshape(-1, 1)
        return -np.mean(y_true * np.log(y_pred + eps) +
                        (1 - y_true) * np.log(1 - y_pred + eps))

    def backward(self, y_true):
        n = self.X.shape[0]
        y_true = y_true.reshape(-1, 1)

        # --- Gradien di output (turunan BCE + sigmoid menyederhanakan jadi ini) ---
        dz2 = (self.a2 - y_true) / n                 # (n, 1)
        dW2 = self.a1.T @ dz2                         # chain rule -> W2
        db2 = dz2.sum(axis=0)

        # --- Propagasi mundur ke hidden layer (chain rule) ---
        da1 = dz2 @ self.W2.T                         # gradien menuju a1
        dz1 = da1 * relu_grad(self.z1)                # lewat turunan ReLU
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0)

        # --- Update parameter (gradient descent) ---
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=2000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(y)
            if epoch % 200 == 0:
                acc = ((y_pred.ravel() > 0.5) == y).mean()
                print(f"Epoch {epoch:4d} | loss={loss:.4f} | acc={acc:.3f}")

    def predict(self, X):
        return (self.forward(X).ravel() > 0.5).astype(int)


def main():
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    nn = NeuralNetwork(n_input=2, n_hidden=8, lr=0.5)
    nn.fit(X_tr, y_tr, epochs=2000)

    test_acc = (nn.predict(X_te) == y_te).mean()
    print(f"\nAkurasi test: {test_acc:.3f}")
    print("(Model 2-layer ini bisa memisahkan data non-linear — "
          "sesuatu yang regresi logistik tunggal tak bisa.)")


if __name__ == "__main__":
    main()
