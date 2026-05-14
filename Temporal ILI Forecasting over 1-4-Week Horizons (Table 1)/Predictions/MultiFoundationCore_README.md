# MultiFoundationCore (MFC): Mixture-of-Experts Forecast Fusion Model

This folder contains the implementation and reproducibility materials for **MultiFoundationCore (MFC)**, a mixture-of-experts forecasting model used for multi-horizon influenza-like illness (ILI) prediction.

MFC is designed to combine the outputs of multiple pretrained or task-specific time series forecasting models. In this implementation, each expert prediction stream is treated as an independent input token, while the observed target/history stream is encoded separately. A cross-attention module then learns how much to rely on each expert stream for 1–4-week-ahead forecasting.

---

## Purpose

The goal of this model is to improve short-term epidemic forecasting by fusing complementary forecasts from several time series models, including PatchTST variants, iTransformer variants, TFT, TiDE, and VanillaTransformer.

The model is evaluated on CDC-aligned short-term forecasting horizons:

```text
1 week ahead
2 weeks ahead
3 weeks ahead
4 weeks ahead
```

The main target in this experiment is regional influenza-like illness (ILI), evaluated across U.S. HHS regions.

---

## Model summary

**MultiFoundationCore** uses a cross-attention fusion architecture:

1. Load prediction files from multiple expert models.
2. Align all experts by region and forecast-origin index.
3. Construct supervised windows from expert predictions and observed target history.
4. Encode each expert stream with an independent LSTM.
5. Encode the observed target/history stream with a separate LSTM.
6. Use the history embedding as the query and expert embeddings as keys/values in multi-head cross-attention.
7. Decode the fused representation into a direct 4-step forecast.

The core design is:

```text
Expert prediction streams  ---> Independent LSTM encoders ----\
                                                                --> Cross-attention fusion --> Decoder --> y(t+1:t+4)
Observed target history ----> History LSTM encoder ------------/
```

The model predicts all four horizons directly rather than recursively rolling one-step predictions forward.

---

## Why this model is useful

Individual forecasting models often perform differently across horizons and epidemic phases. For example, one model may perform well at 1-week-ahead prediction, while another may be more stable at 3–4 weeks ahead. MFC learns a data-driven fusion of these complementary expert signals.

In the paper experiments, MFC improves over the individual expert models by learning when and how to combine their forecasts.

---

## Expected input files

MFC expects one `.pkl` file per expert model. Each file should contain aligned multi-horizon predictions and ground-truth values.

The uploaded experiment uses the following expert prediction files:

```text
PatchTST_predictions.pkl
PatchTSTm_predictions.pkl
PatchTSTt_predictions.pkl
PatchTSTc_predictions.pkl
VanillaTransformer_predictions.pkl
iTransformer_predictions.pkl
iTransformerh_predictions.pkl
PatchTSTh_predictions.pkl
TFT_predictions.pkl
TiDE_predictions.pkl
```

In the uploaded files, each prediction dataframe has shape:

```text
5000 rows × 10 columns
```

with the following columns:

```text
first_ds
unique_id
y_hat_1
y_hat_2
y_hat_3
y_hat_4
y_actual_1
y_actual_2
y_actual_3
y_actual_4
```

Column meaning:

| Column | Description |
|---|---|
| `first_ds` | Forecast-origin index or aligned time index |
| `unique_id` | Region identifier, e.g., `Region 1` |
| `y_hat_1` | Expert prediction for horizon 1 |
| `y_hat_2` | Expert prediction for horizon 2 |
| `y_hat_3` | Expert prediction for horizon 3 |
| `y_hat_4` | Expert prediction for horizon 4 |
| `y_actual_1` | Ground-truth value for horizon 1 |
| `y_actual_2` | Ground-truth value for horizon 2 |
| `y_actual_3` | Ground-truth value for horizon 3 |
| `y_actual_4` | Ground-truth value for horizon 4 |

---

## Main configuration

The paper-style run uses the following configuration:

```python
EPOCHS = 175
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.125

config = MFCConfig(
    prediction_dir="./predictions",
    test_length=200,
    look_back_size=10,
    output_horizon=4,
    expert_lstm_units=16,
    history_lstm_units=16,
    num_heads=8,
    key_dim=16,
    value_dim=16,
    ff_units=16,
    decoder_units=16,
    dropout_rate=DROPOUT_RATE,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    seed=100,
    use_early_stopping=False,
    use_reduce_lr=False,
)
```
---

## Model architecture

The implemented MFC architecture contains:

- independent LSTM encoders for expert prediction streams,
- one LSTM encoder for the observed target/history stream,
- multi-head cross-attention,
- residual connection,
- layer normalization,
- feed-forward fusion block,
- dense decoder for the 4 forecast horizons.

Conceptually:

```text
Input shape:
(batch_size, look_back_size, number_of_expert_features + 1)

Output shape:
(batch_size, output_horizon)
```

For the current experiment:

```text
look_back_size = 10
output_horizon = 4
number_of_expert_models = 10
prediction horizons per expert = 4
expert features = 10 × 4 = 40
history features = 1
total input features = 41
```

---

## Evaluation metrics

The notebook reports the following metrics:

| Metric | Direction |
|---|---|
| MSE | lower is better |
| MAE | lower is better |
| SMAPE | lower is better |
| NNSE | higher is better |
| R² | higher is better |

NNSE is the normalized Nash–Sutcliffe efficiency:

```text
NSE  = 1 - SSE / SST
NNSE = 1 / (2 - NSE)
```

---

## Main MFC results across 5 runs

The notebook reports the following mean performance across five independent runs:

| Metric | Mean | Std. Dev. |
|---|---:|---:|
| MSE ↓ | 0.381649 | 0.007732 |
| MAE ↓ | 0.375617 | 0.002729 |
| SMAPE ↓ | 15.578367 | 0.157813 |
| NNSE ↑ | 0.863655 | 0.002365 |
| R² ↑ | 0.836832 | 0.003317 |

### Cross-run average by horizon

| Horizon | MSE ↓ | MAE ↓ | SMAPE ↓ | NNSE ↑ | R² ↑ |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.118002 | 0.208565 | 8.652771 | 0.949049 | 0.946309 |
| 2 | 0.306495 | 0.336833 | 13.757757 | 0.881752 | 0.865886 |
| 3 | 0.474855 | 0.439076 | 18.179022 | 0.832005 | 0.798074 |
| 4 | 0.627242 | 0.517994 | 21.723917 | 0.791815 | 0.737060 |

---

## Aligned expert baseline summary

The notebook also evaluates individual experts on the same aligned test origins.

| Rank | Expert | Mean MSE ↓ | Mean MAE ↓ | Mean SMAPE ↓ | Mean NNSE ↑ | Mean R² ↑ |
|---:|---|---:|---:|---:|---:|---:|
| 1 | PatchTST-hospitalization | 0.397300 | 0.366674 | 14.950654 | 0.861430 | 0.833429 |
| 2 | PatchTST-covid19 | 0.411609 | 0.377008 | 15.426183 | 0.857090 | 0.827430 |
| 3 | PatchTST-m4 | 0.415453 | 0.375939 | 15.225152 | 0.856087 | 0.825818 |
| 4 | PatchTST | 0.430278 | 0.379108 | 15.458822 | 0.852170 | 0.819603 |
| 5 | PatchTST-trafficl | 0.430833 | 0.379843 | 15.282883 | 0.851639 | 0.819370 |
| 6 | TFT | 0.455457 | 0.415411 | 16.958321 | 0.844492 | 0.809047 |
| 7 | iTransformer | 0.472801 | 0.421137 | 19.129754 | 0.839666 | 0.801775 |
| 8 | iTransformer-hospitalization | 0.474577 | 0.419412 | 18.376175 | 0.838793 | 0.801030 |
| 9 | VanillaTransformer | 0.507529 | 0.433740 | 17.110732 | 0.829919 | 0.787215 |
| 10 | TiDE | 0.588605 | 0.426515 | 16.772222 | 0.810954 | 0.753224 |

MFC achieves better mean MSE and NNSE than the strongest individual expert, showing that the learned cross-attention fusion improves the robustness of the forecast ensemble.

---

