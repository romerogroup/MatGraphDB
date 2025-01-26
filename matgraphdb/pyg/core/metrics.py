import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_correlation_matrix(
    df: pd.DataFrame, columns: list = ["actual", "predicted"]
) -> pd.DataFrame:
    """
    Computes the correlation matrix for the specified columns in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list): List of column names to include in the correlation matrix.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    return df[columns].corr()


def get_pearson_r(
    corr_matrix: pd.DataFrame, col1: str = "actual", col2: str = "predicted"
) -> float:
    """
    Extracts the Pearson correlation coefficient between two specified columns.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        col1 (str): Name of the first column.
        col2 (str): Name of the second column.

    Returns:
        float: Pearson correlation coefficient.
    """
    return corr_matrix.loc[col1, col2]


def compute_r2(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Computes the R² score.

    Args:
        actual (pd.Series): Actual target values.
        predicted (pd.Series): Predicted target values.

    Returns:
        float: R² score.
    """
    return r2_score(actual, predicted)


def compute_mae(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Computes the Mean Absolute Error (MAE).

    Args:
        actual (pd.Series): Actual target values.
        predicted (pd.Series): Predicted target values.

    Returns:
        float: MAE.
    """
    return mean_absolute_error(actual, predicted)


def compute_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Computes the Root Mean Squared Error (RMSE).

    Args:
        actual (pd.Series): Actual target values.
        predicted (pd.Series): Predicted target values.

    Returns:
        float: RMSE.
    """
    return mean_squared_error(actual, predicted, squared=False)