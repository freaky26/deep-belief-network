from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


class BernoulliRBM:
    """A simple Bernoulli RBM trained with contrastive divergence (CD-1)."""

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rng = np.random.default_rng(random_state)

        self.W = self.rng.normal(0.0, 0.1, size=(n_visible, n_hidden))
        self.bv = np.zeros(n_visible, dtype=np.float64)
        self.bh = np.zeros(n_hidden, dtype=np.float64)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-x))

    def sample_hidden(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden_logits = v @ self.W + self.bh
        hidden_probs = self.sigmoid(hidden_logits)
        hidden_states = (self.rng.random(hidden_probs.shape) < hidden_probs).astype(np.float64)
        return hidden_probs, hidden_states

    def sample_visible(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        visible_logits = h @ self.W.T + self.bv
        visible_probs = self.sigmoid(visible_logits)
        visible_states = (self.rng.random(visible_probs.shape) < visible_probs).astype(np.float64)
        return visible_probs, visible_states

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.sigmoid(x @ self.W + self.bh)

    def inverse_transform(self, h: np.ndarray) -> np.ndarray:
        return self.sigmoid(h @ self.W.T + self.bv)

    def fit(self, data: np.ndarray, epochs: int = 20, batch_size: int = 32, verbose: bool = False) -> None:
        n_samples = data.shape[0]

        for epoch in range(epochs):
            permutation = self.rng.permutation(n_samples)
            shuffled = data[permutation]

            for start in range(0, n_samples, batch_size):
                batch = shuffled[start : start + batch_size]

                pos_hidden_probs, pos_hidden_states = self.sample_hidden(batch)

                neg_visible_probs, _ = self.sample_visible(pos_hidden_states)
                neg_hidden_probs, _ = self.sample_hidden(neg_visible_probs)

                batch_size_actual = batch.shape[0]

                positive_grad = (batch.T @ pos_hidden_probs) / batch_size_actual
                negative_grad = (neg_visible_probs.T @ neg_hidden_probs) / batch_size_actual

                self.W += self.learning_rate * (positive_grad - negative_grad) - self.learning_rate * self.weight_decay * self.W
                self.bv += self.learning_rate * (batch.sum(axis=0) - neg_visible_probs.sum(axis=0)) / batch_size_actual
                self.bh += self.learning_rate * (pos_hidden_probs.sum(axis=0) - neg_hidden_probs.sum(axis=0)) / batch_size_actual

            if verbose:
                reconstruction_error = self.reconstruction_error(data)
                print(f"Epoch {epoch + 1}/{epochs} - reconstruction error: {reconstruction_error:.4f}")

    def reconstruction_error(self, data: np.ndarray) -> float:
        reconstructed = self.reconstruct(data)
        return float(np.mean((data - reconstructed) ** 2))

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        hidden = self.transform(data)
        return self.inverse_transform(hidden)


class DeepBeliefNetwork:
    """Stacked RBMs used for unsupervised feature learning."""

    def __init__(self, hidden_layers: list[int]) -> None:
        self.hidden_layers = hidden_layers
        self.rbms: list[BernoulliRBM] = []

    def pretrain(self, data: np.ndarray, epochs: int = 20, batch_size: int = 32, verbose: bool = False) -> np.ndarray:
        current = np.asarray(data, dtype=np.float64)

        for idx, hidden_size in enumerate(self.hidden_layers):
            rbm = BernoulliRBM(
                n_visible=current.shape[1],
                n_hidden=hidden_size,
                learning_rate=0.1,
                random_state=42 + idx,
            )
            rbm.fit(current, epochs=epochs, batch_size=batch_size, verbose=verbose)
            self.rbms.append(rbm)
            current = rbm.transform(current)

        return current

    def encode(self, data: np.ndarray) -> np.ndarray:
        current = np.asarray(data, dtype=np.float64)
        for rbm in self.rbms:
            current = rbm.transform(current)
        return current

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        current = np.asarray(data, dtype=np.float64)
        for rbm in self.rbms:
            current = rbm.transform(current)
        for rbm in reversed(self.rbms):
            current = rbm.inverse_transform(current)
        return current


def plot_feature_visualization(
    features: np.ndarray,
    labels: np.ndarray | None = None,
    output_path: str = "artifacts/feature_visualization.png",
    title: str = "Learned DBN representation",
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(features, dtype=np.float64)
    if X.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, X.shape[0] - 1)))
        X_2d = tsne.fit_transform(X)
    else:
        X_2d = X

    fig, ax = plt.subplots(figsize=(8, 6))

    if labels is not None:
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=18, alpha=0.8)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Class label")
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], s=18, alpha=0.8, color="steelblue")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print("DBN module loaded successfully.")
