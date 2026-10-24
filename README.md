# Breast ultrasound image classification

A compact deep-learning portfolio project that classifies low-resolution breast ultrasound images from the BreastMNIST benchmark as **benign/normal** or **malignant**.

## Approach

The project downloads the official BreastMNIST split (28×28 grayscale images) and trains a small multilayer perceptron with two hidden layers. The dataset's provided train/validation/test split is respected: the held-out test images are not used for model selection.

The model is deliberately small and uses early stopping, making this a quick, reproducible learning project rather than a compute-heavy clinical system.

## Important limitation

This is a benchmark exercise, not a medical device. Its predictions must not be used for diagnosis, screening, or clinical decisions. The low-resolution dataset is limited and does not represent real-world clinical deployment.

## Run

```bash
python -m pip install -r requirements.txt
python src/train_model.py
```

The script saves test metrics, a classification report, the fitted model, a training curve, a confusion matrix, and sample predictions.

## Data source

BreastMNIST is part of [MedMNIST v2](https://medmnist.com/), a collection of standardised biomedical image benchmarks.
