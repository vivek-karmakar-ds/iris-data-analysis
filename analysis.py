"""Explore the Iris dataset and fit a simple classification baseline."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("outputs")
PLOT_PATH = OUTPUT_DIR / "iris_petal_measurements.png"


def main() -> None:
    iris = load_iris(as_frame=True)
    features = iris.data
    target = iris.target
    label_by_number = dict(enumerate(iris.target_names))

    data = features.copy()
    data["species"] = target.map(label_by_number)

    print(f"Rows: {len(data)}")
    print(f"Features: {len(iris.feature_names)}")
    print("\nSamples per species:")
    print(data["species"].value_counts().sort_index())
    print(f"\nMissing values: {int(data.isna().sum().sum())}")
    print(f"Duplicate measurement rows: {int(features.duplicated().sum())}")
    print("\nMean measurements by species:")
    print(data.groupby("species")[iris.feature_names].mean().round(2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_column = "petal length (cm)"
    y_column = "petal width (cm)"

    fig, ax = plt.subplots(figsize=(8, 5))
    for species in iris.target_names:
        subset = data.loc[data["species"] == species]
        ax.scatter(
            subset[x_column],
            subset[y_column],
            label=species,
            alpha=0.8,
        )

    ax.set(
        title="Petal measurements by Iris species",
        xlabel="Petal length (cm)",
        ylabel="Petal width (cm)",
    )
    ax.legend(title="Species")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)
    print(f"\nChart saved to: {PLOT_PATH}")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1_000),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"\nHeld-out accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions))
    print("\nPer-class report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
