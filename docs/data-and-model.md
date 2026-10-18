# Data and Model

[Documentation overview](README.md)

## Dataset files

The project uses the UCI Heart Disease dataset. The exploration notebook downloads `redwankarimsony/heart-disease-data` through `kagglehub`.

| File | Purpose |
| --- | --- |
| `data/raw/heart_disease_uci.csv` | Raw dataset |
| `data/processed/heart_disease_hungary_cleaned.csv` | Client 1 data |
| `data/processed/heart_disease_switzerland_cleaned.csv` | Client 2 data |
| `data/processed/heart_disease_cleveland_cleaned.csv` | Server evaluation data |

Each file has a corresponding `.dvc` pointer containing its checksum, size, and path. The remote is configured in [`.dvc/config`](../.dvc/config) with the S3-compatible endpoint `https://dagshub.com/attmhd/fl-heart-disease.s3` and bucket URL `s3://dvc`.

## Preprocessing notebook

[`eda_uci_heart_disease.ipynb`](../notebook/eda_uci_heart_disease.ipynb) contains data loading, exploration, correlation analysis, partitioning, preprocessing, and CSV export.

The notebook selects these columns in order:

```text
oldpeak, thalch, exang, cp, sex, age, slope, chol, fbs, num
```

The first nine columns are features. `num` is the target.

For each location, the notebook performs the following steps:

1. Select rows belonging to Hungary, Switzerland, or Cleveland.
2. Select the configured feature and target columns.
3. Fill missing numerical values with the column mean and categorical values with the column mode.
4. Apply `LabelEncoder` to the categorical columns identified from the Hungary partition.
5. Convert `num` to binary labels: `0` for no disease and `1` for values of at least one.
6. Apply `StandardScaler` to the numerical feature columns identified from the Hungary partition.
7. Export each processed partition as a CSV file in the notebook's working directory.

Encoding and scaling are fitted separately for each partition. The exported filenames match the three processed filenames listed above.

## Runtime data loading

[`load_data()`](../src/data/dataset.py) reads a processed CSV and returns `X_train`, `X_test`, `y_train`, and `y_test`.

| Operation | Implementation |
| --- | --- |
| Feature extraction | All columns except `num`, preserving their CSV order |
| Target conversion | `(num > 0).astype(int)` |
| Evaluation split | `test_size=0.2` |
| Random seed | `random_state=42` |

Clients use both splits for training and evaluation. The server uses the Cleveland evaluation split for centralized evaluation. Runtime loading uses the feature values already stored in the processed CSV.

## Logistic Regression

[`get_lr_model()`](../src/models/model.py) creates a scikit-learn `LogisticRegression` estimator. The model uses `warm_start=True` and `random_state=42`.

| Setting | Default |
| --- | --- |
| Solver | `saga` |
| Maximum iterations per local fit | `10` |
| Inverse regularization strength, `C` | `1.0` |
| Penalty setting | `l2` |

The factory maps `MODEL_PENALTY=l1` to `l1_ratio=1.0`, uses `l1_ratio=0.0` for `l2`, and sets `C` to infinity for `none`. Solver and regularization settings are read from environment variables.

## Parameter exchange

Model parameters consist of the coefficient array, `coef_`, and the intercept array, `intercept_`. `model_utils.py` extracts these arrays for Flower and assigns received arrays to the local estimator. Assigned models use the binary class labels `[0, 1]`.

During each local fit, the client applies the received global parameters and fits its estimator on the full local training split. The client returns the updated arrays, the number of training samples, and its local metrics.
