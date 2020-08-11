import numpy as np

PARAMS = {
    "changepoint_prior_scale": np.linspace(0.1, 1, 5),
    "yearly_seasonality": [False],
    "daily_seasonality": [False],
}
