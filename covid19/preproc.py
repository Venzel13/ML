import pandas as pd


def preprocess_data(path):
    """
    Preprocess dataset stored in the following `path`

    Parameters:
    ----------
    path: str
        path to the dataset directory.

    Return:
    ------
    df: pandas DataFrame
        Preprocessed dateset.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df.fillna("None", inplace=True)
    df = (df.query("Target == 'ConfirmedCases'")
            .query("TargetValue >= 0"))
    df.drop(["Id", "Target"], axis=1, inplace=True)
    df.rename(columns={"TargetValue": "Infected"}, inplace=True)
    df.columns = map(str.lower, df.columns)

    return df