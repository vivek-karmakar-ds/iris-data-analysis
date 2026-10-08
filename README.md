# Iris Data Analysis & Baseline Classification

A beginner-friendly, reproducible case study using the Iris dataset bundled with scikit-learn. It combines basic data checks, group summaries, a chart, and a simple logistic-regression baseline.

This is a learning project: run the analysis, inspect the outputs, and add your own interpretation before presenting it as completed portfolio work.

## Question

How do sepal and petal measurements vary across the three Iris species, and how well does a simple baseline distinguish them on a held-out sample?

## Dataset

The script loads the classic Iris dataset through scikit-learn, so there is no separate download. It contains 150 samples, 3 classes with 50 samples each, and 4 numeric measurements. See the [scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).

## What the analysis does

- Reports dataset dimensions, class counts, missing values, duplicate measurement rows, and per-species averages.
- Saves a scatter plot of petal length against petal width to `outputs/iris_petal_measurements.png`.
- Fits a scaled logistic-regression baseline using a stratified 80/20 train-test split.
- Prints held-out accuracy, a confusion matrix, and a per-class classification report.

The script prints the metrics when you run it; this README does not assume or promise a particular score.

## Run it

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the packages and run the script:

```bash
python -m pip install -r requirements.txt
python analysis.py
```

## How to interpret the output

Start by comparing the class counts and per-species averages. Then inspect the saved chart to see how the petal measurements overlap. Use the confusion matrix and per-class report to understand which classes the baseline distinguishes or confuses.

## Limitations

This is a small, curated teaching dataset, not a sample of real-world field conditions. A single train-test split is a useful first exercise but is not a robust estimate of generalization. Treat the model as a learning baseline, not a deployable flower-identification system.

## Next steps

- Add a short findings section after you run the script and interpret its output.
- Compare another simple model using the same split.
- Try cross-validation and explain how its estimate differs from one split.
