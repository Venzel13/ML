from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import holidays
import numpy as np
import pandas as pd
from fbprophet import Prophet
from sklearn.model_selection import ParameterGrid

from metrics import smape


def rename_cols(ts, inverse=False):
    """
    Rename `date` and `infected` column to `ds` and `y` respectively and vice
    versa. It's obligatory for fbprophet forecasting.

    Parameters:
    ----------
    ts: pandas Series or DataFrame
        Time series with infected values and date index for a particular
        country/state/county.
    inverse: bool, Optional
        If True, make the opposite transformation to original names.

    Return:
    ------
    renamed: pandas Series or DataFrame
        Renamed ts for a particular country.
    """
    if inverse:
        renamed = ts.rename(
            columns={"ds": "date", "yhat": "pred_infected", "y": "infected",}
        )
    else:
        renamed = ts.rename(columns={"date": "ds", "infected": "y",})

    return renamed


def split_ts(ts, cutoffs):
    """
    Split data into three sets: train/validation/test respectively.
    
    Parameters:
    ----------
    ts: pandas Series or DataFrame
        Time series with infected values and date index for a particular
        country/state/county.
    cutoffs: Iterable
        List of dates with threshold dates for the beginning of the valid and
        test periods.
        
    Return:
    ------
    train/valid/test: pandas Series or DataFrame
        Splitted dataframes for the particular sets.
    """
    train = ts.query("ds < @cutoffs[0]")
    valid = (ts.query("ds >= @cutoffs[0]")
               .query("ds < @cutoffs[1]"))
    test = ts.query("ds >= @cutoffs[1]")

    return train, valid, test


# def find_best_params(train, valid, params):
#     grid = ParameterGrid(params)
#     metrics = np.array([])

#     for param in grid:
#         model = Prophet(**param)
#         model.fit(train)
#         pred = model.predict(valid)["yhat"]
#         metric = smape(valid["y"].values, pred.values)
#         metrics = np.append(metrics, metric)

#     best_params = grid[np.argmin(metrics)]

#     return best_params


def find_best_params(train, valid, params):
    """
    Find the best hyperparameters values among all possible combinations of
    `params`, founded on the validation set. THe proccess is parallelized onto
    all available CPU.
    
    Parameters:
    ----------
    train/valid: pandas Series or DataFrame
        train and validation sets for a particular country/state/county.
    params: dict
        Dictionary of the parameters of interest, where each key represents
        parameter name for Prophet forecaster and each dict value represents
        list of possible values.
        
    Return:
    ------
    best_params: dict
        Dict of best hyperparameters
    """
    grid = ParameterGrid(params)

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        metrics = executor.map(
            iterate_param,
            [train] * len(grid),
            [valid] * len(grid),
            [param for param in grid],
        )

    metrics = list(metrics)
    metrics = np.array(metrics)
    best_params = grid[np.argmin(metrics)]

    return best_params


def iterate_param(train, valid, param):
    """
    Compute sMAPE value for a particular set of parameters `param`.

    Parameters:
    ----------
    train/valid: pandas Series or DataFrame
        train and validation sets for a particular country/state/county.
    param: dict
        Dictionary of the parameters of interest, where each key represents
        parameter name for Prophet forecaster and each dict value represents
        a particular possible parameter from the parameter list.
    
    Return:
    ------
    metric: float,
        The value of sMAPE for a particular set of parameters.
    """
    model = Prophet(**param)
    model.fit(train)
    pred = model.predict(valid)["yhat"]
    metric = smape(valid["y"].values, pred.values)

    return metric


def learn(train, valid, test, best_params):
    """
    Learn algorithm (for a particular country/state/county) on the `train` and
    `valid` sets with the best hyperparameter values
    and predict on the `test` set.
    
    Parameters:
    ----------
    train/valid/test: pandas Series or DataFrame
        train and validation sets for a particular country/state/county.
    best_params: dict
        Dict of best hyperparameters.
        
    Return:
    ------
    pred: pandas Series or DataFrame
        prophet prediction of the test period for a particular country/state/county.

    """
    train = pd.concat([train, valid])

    pred_cols = ["ds", "yhat"]
    country = train["country_region"].values[0]

    model = Prophet(**best_params)

    ### add holiday as a feature if `holiday` package contains a particular country
    if country in dir(holidays.countries):
        model.add_country_holidays(country)

    model.fit(train)
    pred = model.predict(test)[pred_cols]

    return pred


def predict_ts(ts, cutoffs, params):
    """
    Parameters:
    ----------
    ts: pandas Series or DataFrame
        Time series with infected values and date index for a particular
        country/state/county.
    cutoffs: Iterable
        List of dates with threshold dates for the beginning of the valid and
        test periods.
    params: dict
        Dictionary of the parameters of interest, where each key represents
        parameter name for Prophet forecaster and each dict value represents
        list of possible values.
        
    Return:
    ------
    result: pandas Series or DataFrame
        true and predicted values on the test period for a particular
        country/state/county.
    """
    ts = rename_cols(ts)
    train, valid, test = split_ts(ts, cutoffs)
    best_params = find_best_params(train, valid, params)
    pred = learn(train, valid, test, best_params)

    result = pd.concat([train, valid, test])
    result = result.merge(pred, how="outer")
    result = rename_cols(result, inverse=True)

    return result
