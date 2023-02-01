#!/usr/bin/env python3

# Trying out btrack for tracking pollen tubes.
# https://github.com/quantumjot/BayesianTracker

import pandas as pd
from pathlib import Path
import btrack


def load_tsv(file_name: str) -> pd.DataFrame:
    """
    Load a tsv file into a pandas dataframe, based on the location of the 
    python script. The tsv is in a directory called data that is in the same 
    location as the python script.

    Parameters:
    ----------
    file_name : str
        tsv file name

    Returns:
    -------
    pd.DataFrame
        tsv file loaded into a pandas dataframe
    """
    # Get the path of the current script
    current_script_path = Path(__file__).resolve()

    # Navigate to the parent directory
    parent_dir = current_script_path.parent.parent

    # Navigate to the data directory
    data_dir = parent_dir / "data"

    # Get the path of the tsv file
    tsv_file = data_dir / file_name

    # Load the tsv file into a Pandas DataFrame
    df = pd.read_csv(tsv_file, sep='\t')

    return df


def subset_by_confidence_score(df: pd.DataFrame, confidence_score: float) -> pd.DataFrame:
    """
    Subset a dataframe based on a confidence score.

    Parameters:
    ----------
    df : pd.DataFrame
        input dataframe
    confidence_score : float
        minimum score to filter the dataframe

    Returns:
    -------
    pd.DataFrame
        subsetted dataframe
    """
    subset_df = df[df["score"] >= confidence_score]
    return subset_df


def calculate_centroid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the centroid value for each row of a dataframe with columns 
    titled "ymin", "xmin", "ymax", "xmax".

    Parameters:
    ----------
    df : pd.DataFrame
        input dataframe

    Returns:
    -------
    pd.DataFrame
        dataframe with added columns 'centroid_x' and 'centroid_y'
    """
    df["centroid_x"] = df.apply(lambda row: (row["xmin"] + row["xmax"]) / 2, axis=1)
    df["centroid_y"] = df.apply(lambda row: (row["ymin"] + row["ymax"]) / 2, axis=1)
    return df

#def add_rows_to_btrack(df: pd.DataFrame) -> list[PyTrackObject]:
def add_rows_to_btrack(df: pd.DataFrame) -> list:

    """
    Create btrack PyTrackObjects from a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with calculated centroids and confidence score cutoffs

    Returns
    -------
    list
        list of PyTrackObjects, one for each row of the input DataFrame
    """

    btrack_objects = []
    id_counter = 0

    for row in df.itertuples(index=False):
        row_dict = {}
        row_dict['ID'] = id_counter
        row_dict['t'] = getattr(row, "timepoint")
        row_dict['x'] = getattr(row, "centroid_x")
        row_dict['y'] = getattr(row, "centroid_y")
        row_dict['z'] = 0.
        #row_dict['rpn_score'] =
        #row_dict['rpn_obj_type'] =
        #row_dict['rpn_box'] =

        btrack_objects.append(row_dict)
        id_counter += 1

    return btrack_objects


def main():
    df = load_tsv("2022-01-05_run1_26C_D2_t082_stab_predictions.tsv")
    print(df)
    df = subset_by_confidence_score(df, 0.35)
    print(df)
    df = calculate_centroid(df)
    print(df)
    btrack_objects = add_rows_to_btrack(df)
    print(btrack_objects)


if __name__ == "__main__":
    main()
