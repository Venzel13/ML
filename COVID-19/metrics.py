import numpy as np


def smape(T, P):
    """
    Compute symmetric mean absolite percentage error aka sMAPE (see the link:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)

    Parameters:
    ----------
    T: numpy 1-d ndarray
        True labels.
    P: numpy 1-d ndarray
        Predicted labels.

    Return:
    ------
    smape: float
        sMAPE metric value (the best is 0 and the worst is 200%).
    """
    smape = 100 / len(T) * np.sum(2 * np.abs(P - T) / (np.abs(T) + np.abs(P)))

    return smape
