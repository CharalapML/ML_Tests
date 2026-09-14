# Seismic Signal Derivatives — Kurtosis, Auto/Cross-Correlation, Gain Functions, Cepstrum and Array Build

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CharalapML/ML_Tests/blob/main/Seismic_Signal_Derivatives/Seismic_Signal_Derivatives_Kurtosis_Auto_and_XcorrelationsGainFuncsCepstrum_and_array_build.ipynb)

This notebook builds a set of attributes ("derivatives") from raw seismic traces. The
attributes are intended as inputs to a machine-learning routine for identifying the
depth phase within a seismic signal.

The work was carried out while volunteering at the International Seismological Centre
(ISC), with thanks to Dr Tom Garth for guidance and suggestions.

## Contents

The notebook is organised into the following sections:

1. **Load packages** — installs `obspy` and `plotly`, mounts Google Drive.
2. **Load data** — reads BHZ traces (e.g. `II.ARU.00.BHZ`, `IC.HIA.00.BHZ`) from the
   `ISC_RAWFILES_2ndTranche` folder, applies a spline detrend and normalises each trace.
3. **Kurtosis (and skew) functions** — sliding-window kurtosis and its derivatives, used to
   pick the signal onset.
4. **F1max position and signal onset** — locates the characteristic-function maximum and
   defines the onset time.
5. **Initial cross- and auto-correlations** — correlations of the normalised traces and of a
   3-second (60-sample) impulse taken from the onset.
6. **First batch run** — runs the above across the full station list.
7. **Second auto/cross-correlations based on the signal impulse** — applies the impulse
   cross-correlation along each trace from the onset, including a between-signal variant.
8. **Second batch run**
9. **Cepstrum** — cepstral analysis of the traces, including an initial (incomplete)
   windowed-cepstrum comparison.

The kurtosis picker follows:

> C. Baillard, W. Crawford, V. Ballu, C. Hibert and A. Mangeney, *An Automatic
> Kurtosis-Based P- and S-Phase Picker Designed for Local Seismic Networks*, Bulletin of
> the Seismological Society of America, Vol. 104, No. 1, pp. 394–409, February 2014.

## Data

The input traces are not included in this repository. The notebook expects them under
Google Drive at `My Drive/ISC_RAWFILES_2ndTranche/` and mounts the drive via
`google.colab`. To run locally, replace the `siglist` paths in the **Load data** cell with
local file paths and remove the `drive.mount` call.

## Running

The notebook was written for Google Colab — click the badge above to open it there.

To run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Notes

The code has deliberately not been "cleaned" — intermediate plots and exploratory steps
are retained as they were built, since they were useful in developing the approach.
Suggestions for improvement are welcome.
