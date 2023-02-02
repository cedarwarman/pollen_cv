#!/usr/bin/env python3
import btrack.btypes
# Trying out btrack for tracking pollen tubes.
# https://github.com/quantumjot/BayesianTracker

from typing import List
import pandas as pd
import numpy as np
from skimage.io import imread
from pathlib import Path
import glob
import os
# import btrack
from btrack.btypes import PyTrackObject
import napari


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


def subset_df(df: pd.DataFrame, confidence_score: float) -> pd.DataFrame:
    """
    Subset a dataframe based on a confidence score and class.

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
    subset_df = df[(df["score"] >= confidence_score) & (df["class"] == "tube_tip")]
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

    # Coordinates are scaled. Returning to the original dimensions here.
    original_x = 2048
    original_y = 2048

    df["centroid_x"] = df.apply(lambda row: (row["xmin"] + row["xmax"]) / 2 * original_x, axis=1)
    df["centroid_y"] = df.apply(lambda row: (row["ymin"] + row["ymax"]) / 2 * original_y, axis=1)
    return df

#def add_rows_to_btrack(df: pd.DataFrame) -> list[PyTrackObject]:
def add_rows_to_btrack(df: pd.DataFrame) -> List[PyTrackObject]:
    """
    Create btrack PyTrackObjects from a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe with calculated centroids and confidence score cutoffs

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

        obj = PyTrackObject.from_dict(row_dict)
        btrack_objects.append(obj)
        id_counter += 1

    return btrack_objects

def run_tracking(btrack_objects: btrack.btypes.PyTrackObject) -> list:
    """
    Run btrack on a list of btrack objects (PyTrackObject).

    Parameters
    ----------
    btrack_objects : list[PyTrackObject]
        a list of PyTrackObjects made with the add_rows_to_btrack function

    Returns
    -------
    data :

    properties :

    graph :

    """

    with btrack.BayesianTracker() as tracker:
        tracker.max_search_radius = 100
        tracker.tracking_updates = ["MOTION"]

        # configure the tracker using a config file
        tracker.configure("./config/btrack_config.json")

        # append the objects to be tracked
        tracker.append(btrack_objects)

        # set the tracking volume
        tracker.volume = ((0, 0), (1, 1))

        # track them (in interactive mode)
        tracker.track(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks in a format for napari visualization
        data, properties, graph = tracker.to_napari()

        return data, properties, graph


def main():
    df = load_tsv("2022-01-05_run1_26C_D2_t082_stab_predictions.tsv")
    df = subset_df(df, 0.35)
    df = calculate_centroid(df)
    pd.set_option('display.max_columns', None)
    print(df.head(n=5))
    btrack_objects = add_rows_to_btrack(df)
    # print(btrack_objects[0])
    data, properties, graph = run_tracking(btrack_objects)

    # Viewing with Napari
    print("Opening Napari viewer")
    viewer = napari.Viewer()

    print("Adding images")
    image_series = []
    # for image in os.listdir("/Users/cedar/Desktop/well_D2"):
    for image_path in sorted(glob.glob("/Users/cedar/Desktop/well_D2/*.jpg")):
        image = imread(image_path)
        image_array = np.asarray(image)
        image_series.append(image_array)
    viewer_array = np.asarray(image_series)
    viewer.add_image(viewer_array, scale=(1.0, 1.0, 1.0), name='images')

    print("Adding tracks")
    viewer.add_tracks(
        data,
        properties=properties,
        graph=graph,
        name="Tracks",
        tail_width=5,
        tail_length=1000,
        colormap="hsv",
        blending="Translucent",
        opacity=0.7,
        visible=True
    )

    napari.run()
    print("Done")


if __name__ == "__main__":
    main()
