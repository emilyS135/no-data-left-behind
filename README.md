# No Data Left Behind - Exogenous Variables in Long-Term Forecasting of Nursing Staff Capacity

This repository contains code and data in relation to the paper ***No Data Left Behind: Exogenous Variables in Long-Term Forecasting of Nursing Staff Capacity*** by *Emily Schiller (1), Simon Müller (1), Kathrin Ebertsch (2)* and *Jan-Philipp Steghöfer (1)* accepted at DSAA 2024.

(1) XITASO GmbH IT Software Solutions, Augsburg, Germany  

(2) Department for digitalization and nursing science, Augsburg University Hospital, Augsburg, Germany

## Abstract 
Accurate forecasts of nursing staff capacity have the potential to support shift planners in creating optimal schedules for nursing staff, which is crucial for job satisfaction and quality of care provided in hospitals. Recently presented deep learning methods for long-term time series forecasting (LTSF) show promising results on multiple use cases. However, many state-of-the-art LTSF approaches like PatchTST produce univariate forecasts, neglecting potential correlations between different time series such as nursing staff capacities of multiple wards. In this paper, we compare the performance of several LTSF models, namely TSMixer, TiDE, PatchTST, and LightGBM, in forecasting the nursing staff capacity of a ward in a German hospital. These models are benchmarked against traditional approaches, specifically the ARIMA and Naive Seasonal baselines. Additionally, we assess the impact of including exogenous variables from within the hospital as well as external data sources. Our results show that TSMixer outperforms the other models and baselines by up to 57.40 %, with an MAE of 1.126. We find that including exogenous variables improves the performance of TSMixer and LightGBM. To the best of our knowledge, this study is the first to predict nursing staff capacity. 

## Setup

All dependencies are included in the `pyproject.toml` and can be installed using poetry.

1. install poetry `pip install poetry`
2. install dependencies `poetry install`
3. activate virtual environment `poetry shell`
4. execute experiment, e.g., `python train.py -cn lightgbm-uni`



## How to run 

`python train.py -cn {config_name}`

runs an experiment based on a config in the /configs folder. The `-config-name,-cn` parameter sets the "base" config to use in `hydra.main()`.
Apart from that you can use hydra's [Override Syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) to set pre-configured experiment configs or change specific values.

`python train.py -cn lightgbm-uni` trains a lightgbm model using univariate data only.

...

## Data 

The data and its sources are described in the paper's supplementary material (https://zenodo.org/records/13303613) and can also be downloaded from this link https://zenodo.org/records/11104488.
