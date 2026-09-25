# ImageRecognition — CIFAR-10 CNN with MLflow tracking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CharalapML/ML_Tests/blob/main/ImageRecognition/CIFAR10_CNN_MLflow.ipynb)

A Keras `Sequential` CNN that classifies CIFAR-10 images into the ten standard classes,
with every run tracked in MLflow — parameters, metrics, the training-curve figure and
the saved model. Runs in Google Colab or locally; the setup cell detects which and
chooses its output paths accordingly.

Derived from `Class_OOP_CIFAR-10 Run_Image Classification with CNN.ipynb`, restructured
for repeatable use (see [Changes from the original](#changes-from-the-original)).

## What the notebook does

1. **Setup** — detects Colab, mounts Drive there, and creates the output directory:
   - Colab: `MyDrive/ImageRecognition/`
   - local: `./outputs/`, or `$IMGCLASS_OUTPUT_DIR` if set

   Figures, the saved model and the MLflow SQLite database all live under it.
2. **Install** — `mlflow>=3,<4`. TensorFlow/Keras are left as Colab provides them.
3. **Configure** — one `CONFIG` dict holds every tunable: subset sizes, validation
   split, batch size, epochs, learning rate, L2 weight decay and the random seed.
   Nothing below the configuration cell needs editing for a normal run.
4. **`ImgClassification.train()`** — loads the data, previews 64 training images,
   builds and fits the model, then opens a single MLflow run that logs the parameters,
   training curves, test metrics, the sample prediction and the model file.
5. **Inspect** — the last cell reads the runs straight back out with
   `mlflow.search_runs` so you can compare them without leaving the notebook.

## Model

Input `(32, 32, 3)`, four `Conv2D` blocks with batch normalisation and increasing
dropout, then a softmax over ten classes — 1 186 346 parameters:

```
Conv2D(32)  x2 + BatchNorm -> MaxPool(2,2) -> Dropout(0.2)
Conv2D(64)  x2 + BatchNorm -> MaxPool(2,2) -> Dropout(0.3)
Conv2D(128) x2 + BatchNorm -> MaxPool(2,2) -> Dropout(0.4)
Conv2D(256) x2 + BatchNorm -> MaxPool(2,2) -> Dropout(0.5)
Flatten -> Dense(10, softmax)
```

Every `Conv2D` carries an L2 kernel penalty (`weight_decay=0.0001`). Adam at
`lr=0.0005`, categorical cross-entropy, with `ReduceLROnPlateau` and `EarlyStopping`
on `val_loss`.

## Data

CIFAR-10, downloaded automatically by `keras.datasets.cifar10` — nothing to fetch by
hand and nothing dataset-shaped is tracked in this repository.

The defaults use a **subset**: `size_train=4500` and `size_test=500`. Keras holds out
10% of the training subset for validation, so a default run fits 4050 images, validates
on 450 and tests on 500.

> **This subset is too small for meaningful accuracy.** With 4500 images and 11 epochs
> the model lands near chance (~10%), and batch-normalisation running statistics
> estimated from so few batches make the test score noisier still. The architecture
> itself reaches ~85–88% on the full dataset. For real results set
> `size_train=50000, size_test=10000` and `epochs=50` in `CONFIG` — roughly 15–25
> minutes on a Colab GPU.

## Experiment tracking

Runs go to a SQLite backend, which is MLflow's supported store:

```python
TRACKING_URI = f"sqlite:///{OUTPUT_DIR / 'mlflow.db'}"
```

The older file store (`file:///...` or `./mlruns`) is in maintenance mode as of MLflow
3.x and raises `MlflowException` unless `MLFLOW_ALLOW_FILE_STORE=true` is set. To browse
the runs:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
```

On Colab the database is on Drive at `MyDrive/ImageRecognition/mlflow.db`; copy it and
the `mlartifacts/` folder beside it down to run the command above.

To migrate runs from an existing file store, use `mlflow migrate-filestore` — it
converts losslessly.

## Running

Click the Colab badge above, or locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook CIFAR10_CNN_MLflow.ipynb
```

## Changes from the original

Behavioural changes, each deliberate:

- **MLflow backend** is SQLite rather than `file:///content/gdrive/...`, so the
  `MLFLOW_ALLOW_FILE_STORE` opt-out is gone. Runs logged by the old notebook are in a
  separate store and will not appear alongside new ones until migrated.
- **The duplicate `train_test_split` is gone.** The original split the training data
  twice — once in `Data2input`, again in `Data2Use` — shrinking a 4500-image subset to
  3645 train / 405 validation. Validation is now Keras' `validation_split`, giving
  4050 / 450.
- **Normalisation statistics come from the fitted portion only.** `validation_split`
  takes the tail of the training set, so mean and standard deviation are computed from
  everything before that slice instead of the whole array.
- **Callback patience is scaled to the epoch budget.** The original used `patience=10`
  for `ReduceLROnPlateau` and `patience=40` for `EarlyStopping` inside an 11-epoch run,
  so neither could ever fire.
- **The training figure is saved before `plt.show()`.** `show()` closes the figure, so
  the original `plt.savefig` after it wrote a blank image. The hardcoded
  `ColabNotebooks/M_Nadine/epochs/` path is replaced by the configured output directory,
  which is created if missing.
- **Parameters are logged at the start of the run**, not midway through `plots()`, and
  `optimizer` records the class name and learning rate rather than the `repr` of an
  Adam object. The unused `numImage` parameter is dropped.
- **The trained model is logged** as a `.keras` artifact on every run.
- **`warnings.filterwarnings('ignore')` is removed.** It was masking real warnings; the
  metric calls that caused them now pass `zero_division=0` explicitly.
- **One `predict` call on the test set** instead of two, and labels are flattened with
  `.ravel()` before the sklearn metrics.
- **`Input(shape=...)` as the first layer** instead of `input_shape=` on the first
  `Conv2D`, which Keras 3 deprecates.
- **Runs outside Colab.** `google.colab` is imported only when present, and the
  scratch/commented-out cells from the original are removed.

Method names on `ImgClassification` (`Data2input`, `Data2Use`, `ClassCNN`, `modelfit`,
`runMLflow`, `computemetrics`, `plots`, `train`) are unchanged; `evaluate` and
`predict_sample` were split out of `plots`, which previously also evaluated the test
set, downloaded an image, predicted it and logged parameters.

## Notes

- `predict_sample` fetches a reference truck image from GitHub. If the download fails
  the run continues and tags itself `sample_prediction=skipped`, rather than losing a
  completed training run to a network blip.
- `keras.utils.set_random_seed(random_seed)` is called in the constructor. Exact
  reproducibility on GPU still depends on non-deterministic kernels; set
  `TF_DETERMINISTIC_OPS=1` if you need it.
- `validation_split` takes the **last** fraction of the training set before shuffling.
  CIFAR-10 does not arrive sorted by class, so the holdout is representative — but if
  you ever feed in class-ordered data, shuffle first or go back to an explicit split.
