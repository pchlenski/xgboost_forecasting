import anndata
import numpy as np
import pandas as pd


def anndata_to_supervised(
    anndata_obj: anndata.AnnData,
    timestamp_id: str = "timestamp",
    timestamp_type: str = pd.Timestamp,
    subject_id: str = "subject_id",
) -> (np.ndarray, np.ndarray):
    """
    This function turns time series of microbial abundances (or other
    compositional data) into a supervised learning problem.

    Args:
    ----
    anndata (anndata.AnnData): anndata object with adata.obs containing a
        subject_id column and a timestamp column
    timestamp_id (str): name of the column in adata.obs that contains the
        timestamps
    timestamp_type (str): type of the timestamps (default: pd.Timestamp)
    subject_id (str): name of the column in adata.obs that contains the subject
        ids

    Returns:
    -------
    X (np.ndarray): features for supervised learning: time difference plus
        microbial abundances at previous time point
    Y: (np.ndarray): target for supervised learning: microbial abundances at
        current time point
    """

    # Sort dataframe, get shifted values
    df = df.sort_values([subject_id, timestamp_id])
    df_shifted = df.groupby(subject_id).shift(1)

    # Concatenate the original and shifted dataframes. Dropna to remove the first row of each subject
    df_combined = pd.concat([df_shifted, df], axis=1).dropna()

    # Get the values of df_combined as your X and the values of df as your y
    X = df_combined.values
    Y = df.values

    return X, Y
