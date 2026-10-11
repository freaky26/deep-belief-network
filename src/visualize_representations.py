from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

from src.dbn import DeepBeliefNetwork, plot_feature_visualization


def main() -> None:
    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1).astype(np.float64) / 16.0
    labels = digits.target

    dbn = DeepBeliefNetwork(hidden_layers=[32, 16])
    print("Training deep belief network...")
    features = dbn.pretrain(X, epochs=20, batch_size=32, verbose=False)

    output_dir = Path(__file__).resolve().parent.parent / "artifacts"
    output_dir.mkdir(exist_ok=True)

    encoded = dbn.encode(X)
    np.save(output_dir / "learned_features.npy", encoded)

    plot_feature_visualization(
        encoded,
        labels=labels,
        output_path=str(output_dir / "feature_visualization.png"),
        title="Learned DBN feature space",
    )

    print(f"Saved learned features to {output_dir / 'learned_features.npy'}")
    print(f"Saved visualization to {output_dir / 'feature_visualization.png'}")


if __name__ == "__main__":
    main()
