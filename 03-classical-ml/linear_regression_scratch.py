"""
Regresi Linear dari NOL — hanya dengan NumPy.

Tujuan: memahami apa yang sebenarnya dilakukan `LinearRegression().fit()` di balik
layar. Kita implementasikan gradient descent (Modul 00.3) untuk meminimalkan MSE.

Jalankan:  python linear_regression_scratch.py
"""
import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionScratch:
    """Regresi linear dilatih dengan batch gradient descent."""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None      # bobot per fitur (w)
        self.bias = None         # intercept (b)
        self.loss_history = []   # untuk memantau konvergensi

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 1. Inisialisasi parameter dengan nol
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # 2. Loop gradient descent
        for i in range(self.n_iterations):
            # --- Forward: prediksi ---
            y_pred = X @ self.weights + self.bias        # (n_samples,)

            # --- Hitung loss (MSE) untuk dipantau ---
            error = y_pred - y
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            # --- Hitung gradien (turunan MSE terhadap w dan b) ---
            # dL/dw = (2/n) * X^T (y_pred - y)
            # dL/db = (2/n) * sum(y_pred - y)
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # --- Update parameter: langkah turun melawan gradien ---
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                print(f"Iter {i:4d} | loss (MSE) = {loss:.4f}")

        return self

    def predict(self, X):
        return X @ self.weights + self.bias


def main():
    # --- Buat data sintetis: y = 4 + 3x + noise ---
    rng = np.random.default_rng(42)
    X = 2 * rng.random((200, 1))                 # fitur tunggal di [0, 2)
    y = 4 + 3 * X[:, 0] + rng.normal(0, 0.5, 200)  # target dengan noise

    # --- Normalisasi fitur (membantu konvergensi GD) ---
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    # --- Latih model dari nol ---
    model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X_norm, y)

    print(f"\nBobot dipelajari : {model.weights}")
    print(f"Bias dipelajari  : {model.bias:.4f}")

    # --- Bandingkan dengan scikit-learn (harus mirip) ---
    from sklearn.linear_model import LinearRegression
    sk = LinearRegression().fit(X_norm, y)
    print(f"\n[scikit-learn] coef={sk.coef_}, intercept={sk.intercept_:.4f}")

    # --- Visualisasi ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.scatter(X[:, 0], y, alpha=0.4, label="data")
    x_line = np.linspace(0, 2, 100).reshape(-1, 1)
    x_line_norm = (x_line - X_mean) / X_std
    ax1.plot(x_line, model.predict(x_line_norm), "r-", lw=2, label="prediksi")
    ax1.set_title("Garis hasil belajar"); ax1.legend()

    ax2.plot(model.loss_history)
    ax2.set_xlabel("iterasi"); ax2.set_ylabel("MSE")
    ax2.set_title("Loss turun seiring training (gradient descent)")

    plt.tight_layout()
    plt.savefig("linear_regression_result.png", dpi=100)
    print("\nPlot disimpan ke linear_regression_result.png")


if __name__ == "__main__":
    main()
