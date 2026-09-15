# 60_NN — Individual Cubes, Individual Attributes (20 Jan 2022)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CharalapML/ML_Tests/blob/main/60_NN_IndividualCubes/60_NN_IndividualCubes_Individual_Atrributes_20_01_2022.ipynb)

A 3D convolutional encoder–decoder (U-Net-style, with skip connections) that learns to
predict a target trace from a single seismic-signal attribute. This is one of the "CNN
Final / Semi Final" experiments from the ISC 2020 work: the input "cube" is split into
its individual attributes and the network is trained on **one attribute at a time** to
see how each contributes to locating the depth phase.

The attributes were produced by the companion notebook in
[`../Seismic_Signal_Derivatives`](../Seismic_Signal_Derivatives).

## What the notebook does

1. **Load data** — mounts Google Drive and loads two `.npy` arrays from
   `My Drive/CNN_TestData/`:
   - `4Darray_2_Ordered_in.npy` — input, shape `(events, receivers, attributes, samples)`
     = `(4, 10, 5, 8000)`
   - `4Darray_2_Ordered_out.npy` — target, shape `(4, 10, 1, 8000)`

   NaNs in the target are zeroed. (The `4Darray_1_Unordered_*` pair is left commented
   out as an alternative.)
2. **Split the input cube into individual attributes** along axis 2:

   | index | attribute |
   |------:|-----------|
   | 0 | Original splined, scaled signal |
   | 1 | `F1cf` |
   | 2 | `F4cf` |
   | 3 | `F4cfgrad` |
   | 4 | `F4cfgrad_gtmean` |

   Each is given a trailing channel dimension for the CNN. The run as saved trains on
   `F4cfgrad_gtmean`; change the array passed to `train_test_split` to try another.
3. **Train/test split** — `sklearn.model_selection.train_test_split`, `test_size=0.25`.
4. **Model** — Keras functional API, input `(10, 1, 8000, 1)`:
   - Encoder: `Conv3D` → `MaxPool3D` ×3, then a strided `Conv3D`, pooling only along the
     trace-length axis so receivers and attributes are never mixed.
   - Decoder: `Conv3DTranspose` → `Concatenate` (skip from the matching encoder level) →
     `UpSampling3D` ×3, a final `Conv3DTranspose`, then a `MaxPool3D` and a 1×1×1 `Conv3D`
     down to a single output channel.
5. **Compile & fit** — Adam optimiser, mean-squared-error loss, 10 epochs.
6. **Visualise the architecture** — `visualkeras.layered_view` and Keras `plot_model`
   (writes `model_plot.png`).
7. **Inspect predictions** — per-receiver plots of `|prediction|` against the target and
   input trace, with a mean-threshold line and a 20-sample moving average (via
   `bottleneck.move_mean`) to smooth the picked response.

## Data

The `.npy` inputs are **not** in this repository. The notebook expects them under
Google Drive at `My Drive/CNN_TestData/`. To run locally, edit the two `np.load(...)`
paths in the load-data cell and remove the `drive.mount` call.

## Running

Written for Google Colab — click the badge above. To run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

`plot_model` needs the Graphviz binaries installed on the system as well as the
`pydot`/`graphviz` Python packages (`brew install graphviz` on macOS).

## Notes

- The notebook mixes `keras.*` and `tensorflow.keras.*` imports, which was fine on the
  Colab TF/Keras versions of early 2022. On Keras 3 the `keras.utils.vis_utils` import
  should be `keras.utils`.
- Exploratory plots and commented-out alternatives are retained as they were built.
