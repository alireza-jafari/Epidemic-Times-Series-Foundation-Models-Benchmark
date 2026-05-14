# Table I Reproduction: Temporal ILI Forecasting over 1--4-Week Horizons

This folder contains the notebooks and configuration information used to reproduce Table I of the paper. Table I evaluates temporal, within-region forecasting of regional Influenza-Like Illness (ILI) over 1--4-week-ahead horizons. The table compares pattern models, pretrained/foundation models, LLM-style time-series models, and the MultiFoundationCore fusion model.

All reported metrics are computed on the ILI target.

## Experiment Summary

- **Forecasting target:** Influenza-Like Illness (ILI)
- **Spatial unit:** 10 U.S. HHS regions
- **Temporal frequency:** Weekly
- **Evaluation regime:** Temporal within-region forecasting
- **Forecasting horizons:** 1--4 weeks ahead
- **Metrics:** Mean Squared Error (MSE) and Normalized Nash--Sutcliffe Efficiency (NNSE)
- **Primary look-back window:** 52 weeks for single NeuralForecast, Chronos, and LSTM models
- **Rolling retraining window:** 200 forecast origins for NeuralForecast models
- **Output format:** Direct 4-step output for direct models; recursive/iterative decoding for iterative models

## Notebooks

### `Other_NeuralForecast_models.ipynb`

Used for the non-pretrained NeuralForecast baselines, including:

- TCN
- TiDE
- VanillaTransformer
- TimeLLM
- TimesNet
- TFT

The model name, learning rate, training steps, look-back size, and retraining window should be set according to the configuration table below.

### `PatchTST-Foundation.ipynb`

Used for PatchTST-based Table I experiments. The main Table I configuration is:

- `TB_h4_lr_0.0005_epoch_150_lbs_52_window_retrain_200`

This notebook is also the template for PatchTST variants pretrained on external datasets such as M4, TrafficL, Epidemic, and Hospitalization.

### `iTransformer-Foundation.ipynb`

Used for iTransformer-based Table I experiments. The main Table I configuration is:

- `TB_h4_lr_0.0001_epoch_75_lbs_52_window_retrain_200`

This notebook is also the template for iTransformer variants pretrained on external datasets such as M4, TrafficL, Epidemic, and Hospitalization.

### `PatchTST and Transformer pretrained on Hospitalization.ipynb`

Used for the hospitalization-pretrained PatchTST and iTransformer variants.

### `Chronos-T5.ipynb`

Used for the Chronos-T5 pretrained checkpoint experiments:

- Chronos-T5-mini
- Chronos-T5-base
- Chronos-T5-large

The notebook evaluates pretrained Chronos models with a 52-week context and 4-week prediction horizon. These models are evaluated as pretrained foundation models rather than trained from scratch on ILI.

### `Chronos-Bolt.ipynb`

Used for the Chronos-Bolt pretrained checkpoint experiments:

- Chronos-Bolt-mini
- Chronos-Bolt-small
- Chronos-Bolt-base

The notebook evaluates pretrained Chronos-Bolt models with a 52-week context and 4-week prediction horizon.

### `LSTM-TensorFlow.ipynb`

Used for the two LSTM baselines in Table I:

- `LSTM-direct`: predicts the full 4-week output vector directly.
- `LSTM-iterative`: trains a one-step model and recursively rolls the predicted value into the input window to generate 1--4-week forecasts.

### `FM-MultiFoundationCore.ipynb`

Used for the MultiFoundationCore fusion model. MultiFoundationCore takes forecast-derived expert streams as input and learns a task-specific fusion model to predict the 4-week ILI forecast vector.

## Table I Model Configurations

| Model | Implementation | Strategy | Horizon | Input size | Training steps / epochs | Learning rate | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARIMA | Statsmodels | Iterative | 4 | 104 | -- | -- | -- |
| TCN | NeuralForecast | Direct | 4 | 52 | 400 | 0.001 | MSE |
| TiDE | NeuralForecast | Direct | 4 | 52 | 3000 | 0.001 | MSE |
| VanillaTransformer | NeuralForecast | Direct | 4 | 52 | 500 | 0.001 | MSE |
| TimeLLM | NeuralForecast | Direct | 4 | 52 | 400 | 0.001 | MSE |
| TimesNet | NeuralForecast | Direct | 4 | 52 | 200 | 0.001 | MSE |
| TFT | NeuralForecast | Direct | 4 | 52 | 200 | 0.001 | MSE | 200 |
| iTransformer | NeuralForecast | Direct | 4 | 52 | 75 | 0.0001 | MSE |
| PatchTST | NeuralForecast | Direct | 4 | 52 | 150 | 0.0005 | MSE |
| Chronos-T5 variants | Chronos | Iterative | 4 | 52 | -- | -- | -- |
| Chronos-Bolt variants | Chronos | Iterative | 4 | 52 | -- | -- | -- |
| LSTM-direct | TensorFlow/Keras | Direct | 4 | 52 | 100 | 0.0001 | MSE |
| LSTM-iterative | TensorFlow/Keras | Iterative | 4 | 52 | 100 | 0.0001 | MSE |
| MultiFoundationCore | Custom | Direct | 4 | 10 | 700 | 0.0002 | MSE |

## Exact Configuration Strings

The NeuralForecast experiments are organized by configuration strings with the format:

```text
TB_h{horizon}_lr_{learning_rate}_epoch_{training_steps}_lbs_{look_back_size}_window_retrain_{retrain_window}
```

The exact Table I configuration strings are:

```text
TCN:                TB_h4_lr_0.001_epoch_400_lbs_52_window_retrain_200
TiDE:               TB_h4_lr_0.001_epoch_3000_lbs_52_window_retrain_200
VanillaTransformer: TB_h4_lr_0.001_epoch_500_lbs_52_window_retrain_200
TimeLLM:            TB_h4_lr_0.001_epoch_400_lbs_52_window_retrain_200
TimesNet:           TB_h4_lr_0.001_epoch_200_lbs_52_window_retrain_200
TFT:                TB_h4_lr_0.001_epoch_200_lbs_52_window_retrain_200
iTransformer:       TB_h4_lr_0.0001_epoch_75_lbs_52_window_retrain_200
PatchTST:           TB_h4_lr_0.0005_epoch_150_lbs_52_window_retrain_200
```

## Recommended Run Order

1. Prepare the ILI data using the shared data-generation code.
2. Run the direct NeuralForecast baselines.
3. Run PatchTST and iTransformer pretrained variants.
4. Run the Chronos-T5 and Chronos-Bolt checkpoint evaluations.
5. Run the LSTM direct and iterative baselines.
6. Generate the expert prediction files needed by MultiFoundationCore.
7. Run MultiFoundationCore.
8. Aggregate per-horizon MSE and NNSE into the final Table I format.

## Expected Output Files

Each model should save:

- Per-seed prediction files.
- Per-horizon metric summaries.
- Aggregated mean and standard deviation across seeds.
- Training-loss plots for trainable NeuralForecast models.
- A final results text or CSV file containing:
  - Average MSE over horizons 1--4.
  - Average NNSE over horizons 1--4.
  - Horizon-specific MSE and NNSE for weeks 1, 2, 3, and 4.

## Notes for Repository Cleanup

- Put all hard-coded configuration values into YAML or JSON files.
- Avoid editing model names manually inside notebooks; use a command-line argument or config file.
- Save all result summaries as machine-readable CSV files.
- Fit any normalization only on the training split.
- Keep pretrained-checkpoint evaluations separated from train-from-scratch experiments.
- Store MultiFoundationCore expert prediction files in a dedicated `results/predictions/table1/` directory.
