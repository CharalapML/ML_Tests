# 61_NN — Sequential CNN, Multiple Attributes (18 Sep 2022)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CharalapML/ML_Tests/blob/main/61_NN_Sequential_Multiple_Attributes/61_NN_Sequential_Multiple_Attributes_18_09_22.ipynb)

A compact Keras `Sequential` 3D CNN that takes the **full five-attribute input cube** at
once (rather than one attribute at a time, as in
[`../60_NN_IndividualCubes`](../60_NN_IndividualCubes)) and regresses a flattened target
trace for every receiver. The notebook also includes a small optimiser/loss grid search.

The attributes were produced by
[`../Seismic_Signal_Derivatives`](../Seismic_Signal_Derivatives).

## What the notebook does

1. **Load data** — mounts Google Drive and loads two `.npy` arrays from
   `My Drive/CNN_TestData/`:
   - `4Darray_2_Ordered_in.npy` — input, shape `(events, receivers, attributes, samples)`
     = `(4, 10, 5, 8000)`
   - `4Darray_2_Ordered_out.npy` — target, shape `(4, 10, 1, 8000)`

   NaNs in the target are zeroed. (The `4Darray_1_Unordered_*` pair is left commented
   out as an alternative.)
2. **Reshape the target** — each event's `(10, 1, 8000)` target is flattened to a single
   `80000`-sample vector, giving `target_ys` of shape `(4, 80000)`.
3. **Model** — `Sequential`, input `(10, 5, 8000, 1)`:

   ```
   Conv3D(32, (2,1,2), strides=(2,1,2), relu)
   MaxPooling3D((1,2,2))
   Conv3D(64, (1,1,2), strides=(1,1,2), relu)
   MaxPooling3D((1,2,2))
   Conv3D(64, (1,1,2), strides=(1,1,2), relu)
   Flatten
   ```

   The flattened output is compared directly to the 80 000-sample target — there is no
   final `Dense` layer, so the flattened size must match. Per-layer output shapes are
   printed to check this.
4. **Train** — Adam, MSE loss, 4 epochs, `batch_size=1`, on the first three events;
   the fourth event is held out and scored with `model.evaluate`.
5. **Plot** loss and accuracy curves from `history`.
6. **Optimiser / loss grid search** — loops over `['adam', 'nadam']` ×
   `['mse', 'huber']`, re-compiles and re-fits, and collects the final-epoch accuracy of
   each pairing in `Accuracies` for comparison.
7. **Visualise the architecture** with `visualkeras.layered_view`.

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

## Notes

- Work in progress: the notebook's own to-do list is a fuller grid search over
  optimisers and loss functions and a plot of the results, with the note that accuracy
  is probably the better metric for comparing pairings. Normalising all input traces is
  also flagged but not yet done.
- The grid-search loop re-fits the *same* `model` object each time (weights are not
  reset between pairings), and appends `history.history['accuracy'][-1]` from the
  original `history` rather than the return value of the new `fit` call — so the recorded
  accuracies are all identical. Assign `history = model.fit(...)` inside the loop and
  rebuild the model per pairing to get a true comparison.
- The last two cells are scratch (`reshape(model.layers[-1])` and a pasted
  `np.reshape` signature) and will raise if run.
- Mixes `keras.*` and `tensorflow.keras.*` imports, which was fine on the Colab versions
  of 2022.
