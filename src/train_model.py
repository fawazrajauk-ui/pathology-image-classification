"""Train a small neural network on BreastMNIST ultrasound images."""
from pathlib import Path
from urllib.request import urlretrieve
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "breastmnist.npz"
URL = "https://zenodo.org/records/10519652/files/breastmnist.npz?download=1"

def main():
    for folder in (ROOT / "data", ROOT / "figures", ROOT / "models", ROOT / "results"):
        folder.mkdir(exist_ok=True)
    if not DATA.exists():
        print("Downloading BreastMNIST...")
        urlretrieve(URL, DATA)
    data = np.load(DATA)
    x_train = data["train_images"].reshape(len(data["train_images"]), -1) / 255.0
    y_train = data["train_labels"].ravel()
    x_val = data["val_images"].reshape(len(data["val_images"]), -1) / 255.0
    y_val = data["val_labels"].ravel()
    x_test = data["test_images"].reshape(len(data["test_images"]), -1) / 255.0
    y_test = data["test_labels"].ravel()
    x_fit, y_fit = np.vstack([x_train, x_val]), np.hstack([y_train, y_val])
    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
                          batch_size=64, learning_rate_init=1e-3, max_iter=80,
                          early_stopping=True, validation_fraction=0.15,
                          n_iter_no_change=10, random_state=42)
    model.fit(x_fit, y_fit)
    pred = model.predict(x_test)
    metrics = {"test_accuracy": accuracy_score(y_test, pred), "test_balanced_accuracy": balanced_accuracy_score(y_test, pred), "iterations": model.n_iter_}
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ROOT / "results" / "classification_report.txt").write_text(classification_report(y_test, pred, target_names=["benign/normal", "malignant"]))
    import joblib
    joblib.dump(model, ROOT / "models" / "breastmnist_mlp.joblib")
    plt.figure(figsize=(7,4)); plt.plot(model.loss_curve_); plt.xlabel("Iteration"); plt.ylabel("Training loss"); plt.title("Neural-network training curve"); plt.tight_layout(); plt.savefig(ROOT / "figures" / "training_curve.png", dpi=160); plt.close()
    fig, ax = plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["benign/normal", "malignant"], ax=ax, colorbar=False); ax.set_title("Test-set confusion matrix"); fig.tight_layout(); fig.savefig(ROOT / "figures" / "confusion_matrix.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(10,4));
    for ax, image, truth, guess in zip(axes.ravel(), data["test_images"][:10], y_test[:10], pred[:10]):
        ax.imshow(image, cmap="gray"); ax.set_title(f"true {truth} / pred {guess}"); ax.axis("off")
    fig.suptitle("BreastMNIST held-out predictions"); fig.tight_layout(); fig.savefig(ROOT / "figures" / "sample_predictions.png", dpi=160); plt.close(fig)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
