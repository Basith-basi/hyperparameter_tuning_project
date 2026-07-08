import pandas as pd


def load_data(file_path):
    """
    Loads a CSV file and returns a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    df = pd.read_csv(file_path)

    return df