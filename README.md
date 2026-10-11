# Deep Belief Network for Unsupervised Feature Learning

This project implements a small Deep Belief Network (DBN) built from stacked Restricted Boltzmann Machines (RBMs) for unsupervised feature learning. The model is trained on unlabeled data, and the resulting hidden activations are visualized to reveal the structure of the learned representation.

## What this project does

- Trains a stack of RBMs in an unsupervised way
- Learns useful latent features from unlabeled input
- Encodes the data into a compact representation
- Visualizes the learned features with t-SNE
- Uses a real dataset (`sklearn.datasets.load_digits`) as a demonstration

## Project structure

- `src/dbn.py` — DBN and RBM implementation
- `src/train_dbn.py` — end-to-end training script
- `src/visualize_representations.py` — visualization entry point
- `requirements.txt` — dependencies
- `README.md` — project overview and usage

## Requirements

Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run training

From the repository root:

```bash
python src/train_dbn.py
```

This will:

- load the digits dataset
- train the DBN in an unsupervised manner
- save learned features to `artifacts/learned_features.npy`
- create a visualization at `artifacts/feature_visualization.png`

## Visualize learned representations

```bash
python src/visualize_representations.py
```

This script loads the learned feature map and recreates the t-SNE plot.

## Notes

The model is intentionally lightweight and educational. It demonstrates the core idea of DBNs: layer-wise greedy pretraining with RBMs followed by feature extraction.

This is a good template for learning and extension toward larger datasets or more complex architectures.
