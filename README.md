# Understanding Key Features of Time Series Foundation Models from Epidemic Forecasting

Code, data-processing scripts, model configurations, and reproducibility materials for the paper:

**Understanding Key Features of Time Series Foundation Models from Epidemic Forecasting**

This repository provides a reusable benchmark for evaluating modern time series forecasting models on regional influenza forecasting tasks. The study focuses on U.S. Health and Human Services (HHS) region-level influenza-like illness (ILI) and influenza-associated hospitalization time series, with CDC-aligned 1–4-week-ahead forecasting horizons.

The repository includes implementations and experiment pipelines for classical statistical baselines, deep learning forecasters, numerical time series foundation models, LLM-style time series models, and the proposed mixture-of-experts framework, **MultiFoundationCore**.


## Overview

Accurate short-term epidemic forecasting is important for vaccination planning, hospital staffing, and public-health preparedness. This project evaluates whether modern time series foundation models can improve regional influenza forecasting under realistic public-health constraints.

The main forecasting tasks are:

1. **Influenza-like illness (ILI) forecasting**

   * Weekly ILI time series.
   * 10 U.S. HHS regions.
   * Approximately 20 years of regional surveillance history.
   * Forecasting horizons: 1, 2, 3, and 4 weeks ahead.

2. **Influenza-associated hospitalization forecasting**

   * Weekly region-level hospitalization time series.
   * Approximately three influenza seasons.
   * Used both as a forecasting target and as an auxiliary related-domain signal.

The paper compares temporal within-region forecasting and spatial across-region generalization. It also studies the role of pretraining, auxiliary signals, retraining frequency, and model family differences.

---

## Main research questions

This repository supports experiments addressing the following questions:

* Do time series foundation models transfer effectively to operational influenza forecasting?
* How do numerical time series models compare with LLM-style time series models under CDC-aligned 1–4-week horizons?
* Does pretraining on related domains, such as influenza hospitalizations, improve multi-horizon ILI forecasting?
* How much does spatial distribution shift affect forecasting performance across unseen HHS regions?
* Can a mixture-of-experts design improve robustness by combining multiple pretrained forecasters?
* How does retraining frequency affect temporal adaptation and long-horizon accuracy?

---

## Models included

The repository contains scripts and configurations for the following model families.

### Statistical baseline

* ARIMA

### Deep learning and numerical forecasting models

* LSTM
* TinyLSTM
* TCN
* TFT
* TiDE
* Vanilla Transformer
* TimesNet
* PatchTST
* iTransformer

### Time series foundation and LLM-style models

* Chronos-T5 variants
* Chronos-Bolt variants
* TimeLLM-GPT2

### Proposed / hybrid model

* MultiFoundationCore

MultiFoundationCore is a mixture-of-experts framework that combines predictions or representations from multiple pretrained and task-specific forecasters. It is designed to support auxiliary signals such as influenza-associated hospitalization data and to improve robustness across multi-horizon forecasting tasks.

---

## Datasets

The experiments use the following datasets.

### Evaluation targets

| Dataset                               |        Spatial unit | Frequency | Role                                    |
| ------------------------------------- | ------------------: | --------: | --------------------------------------- |
| Influenza-like illness (ILI)          | 10 U.S. HHS regions |    Weekly | Main forecasting target                 |
| Influenza-associated hospitalizations | 10 U.S. HHS regions |    Weekly | Forecasting target and auxiliary signal |

### Auxiliary / pretraining datasets

| Dataset                    | Role                                                              |
| -------------------------- | ----------------------------------------------------------------- |
| Influenza hospitalizations | Related-domain pretraining and auxiliary input for ILI prediction |
| ILI                        | Related-domain auxiliary input for hospitalization prediction     |
| Epidemic weekly data       | Epidemic-domain pretraining source                                |
| Traffic / TrafficL         | Non-epidemiological structured time series pretraining source     |
| M4                         | Large generic time series pretraining corpus                      |

---

## Forecasting settings

### Temporal evaluation: within-region forecasting

In the temporal setting, each HHS region is split chronologically. Models are trained on earlier observations and evaluated on future observations from the same region. This tests forward-in-time generalization.

### Spatial evaluation: across-region forecasting

In the spatial setting, models are trained on a subset of HHS regions and evaluated on held-out regions. This tests geographic transfer and robustness to regional distribution shift.

### Forecasting horizons

All main forecasting experiments use CDC-aligned short-term horizons:

```text
1 week ahead
2 weeks ahead
3 weeks ahead
4 weeks ahead
```

### Output strategies

The repository supports both:

* **Direct multi-output forecasting:** the model predicts the full 1–4-week trajectory in one forward pass.
* **Iterative forecasting:** the model predicts one step ahead and recursively rolls forward.

---

## Metrics

The main evaluation metrics are:

* **Mean Squared Error (MSE)**
* **Normalized Nash–Sutcliffe Efficiency (NNSE)**

NNSE is used because it provides a normalized performance score. A value near 1 indicates strong predictive agreement, while values near 0.5 correspond to a mean-baseline-like model.

Additional metrics may be computed in some scripts:

* Mean Absolute Error (MAE)
* Symmetric Mean Absolute Percentage Error (SMAPE)
* Root Mean Squared Error (RMSE)
* Coefficient of determination (R²)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Epidemic-Times-Series-Foundation-Models-Benchmark.git
cd Epidemic-Times-Series-Foundation-Models-Benchmark
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate epidemic-tsfm
```

Alternatively, install from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data preparation

Download or place the required raw files under:

```text
data/raw/
```

Expected processed format:

```text
unique_id, ds, y
Region 1, 2003-10-05, ...
Region 1, 2003-10-12, ...
Region 2, 2003-10-05, ...
```

Where:

* `unique_id` is the region identifier.
* `ds` is the weekly date.
* `y` is the target value.


## Main results summary

The paper reports several major findings:

1. **MultiFoundationCore achieves the strongest overall performance** in temporal ILI forecasting by fusing multiple expert forecasters.
2. **PatchTST and iTransformer are strong numerical time series models** for CDC-aligned short-horizon influenza prediction.
3. **Pretraining improves multi-horizon robustness**, especially at longer horizons and especially when the pretraining source is mechanistically related to influenza dynamics.
4. **Hospitalization data is useful in both directions:** it improves ILI forecasting when used for pretraining or as an auxiliary signal, and ILI improves hospitalization forecasting as an auxiliary covariate.
5. **LLM-style time series models underperform strong numerical models** in this operational 1–4-week forecasting setting.
6. **Spatial generalization is harder than temporal forecasting**, because held-out regions introduce distribution shift across geography.
7. **Moderate retraining frequency improves temporal adaptation**, especially at 3–4-week horizons.

---

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{jafari2026epidemic_tsfm,
  title   = {Understanding Key Features of Time Series Foundation Models from Epidemic Forecasting},
  author  = {Jafari, Alireza and Fox, Judy and Fox, Geoffrey C. and Marathe, Madhav and Chou, Jingyuan and Adiga, Aniruddha},
  journal = {Under review},
  year    = {2026}
}
```

---

## License

This project is released under the MIT License unless otherwise specified.

Data sources may have their own licenses or usage terms. Please check the original data providers before redistributing raw datasets.

---

## Contact

For questions, please contact:

```text
Alireza Jafari
University of Virginia
Email: jrp5td@virginia.edu
```

---

## Acknowledgments

This repository accompanies research on time series foundation models for public-health forecasting. The experiments use regional influenza surveillance and hospitalization time series and compare a broad set of statistical, neural, foundation, and mixture-of-experts forecasting methods.
