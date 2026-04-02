from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def train_and_save_model() -> Path:
    x, y = make_classification(
        n_samples=2000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    model = LogisticRegression(max_iter=500)
    model.fit(x, y)

    model_dir = Path(__file__).resolve().parents[1] / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    return model_path


if __name__ == "__main__":
    saved_path = train_and_save_model()
    print(f"Model saved at: {saved_path}")
