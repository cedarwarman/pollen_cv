#!/usr/bin/env python3

"""
Trying out btrack for tracking pollen tubes.

Original btrack repository is located here:
https://github.com/quantumjot/BayesianTracker
"""

from pathlib import Path
from typing import List, Tuple, Union

import btrack
import btrack.btypes
import napari
import numpy as np
import pandas as pd
from skimage.io import imread


def load_tsv(
    file_name: str
) -> pd.DataFrame:
    """Load a tsv file into a pandas dataframe.

    This function uses the location of the python script. The tsv must be in a
    directory called "data" that is in the same location as the python script.

    Parameters:
    ----------
    file_name : str
        The name of the tsv file.

    Returns:
    -------
    df : pd.DataFrame
        The tsv file loaded into a pandas dataframe.

    """

    # Add the rest of the file name based on the inference output convention
    file_name = file_name + "_t082_stab_predictions.tsv"

    # Get the path of the current script
    current_script_path = Path(__file__).resolve()

    # Navigate to the parent directory
    parent_dir = current_script_path.parent.parent

    # Navigate to the data directory
    data_dir = parent_dir / "data" / "btrack_inference"

    # Get the path of the tsv file
    tsv_file = data_dir / file_name

    # Load the tsv file into a Pandas DataFrame
    df = pd.read_csv(tsv_file, sep='\t')

    # Fix a bad column name
    df = df.rename(columns={'class': 'class_label'})

    return df


def subset_df(
    df: pd.DataFrame,
    confidence_score: float,
    class_string: str
) -> pd.DataFrame:
    """Subset a dataframe based on a confidence score and class.

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe.
    confidence_score : float
        Minimum score to filter the dataframe.
    class_string : str
        Class for subsetting the dataframe. Either "tip" or "pollen"

    Returns:
    -------
    subsetted_df : pd.DataFrame

    """

    if class_string == "tip":
        subsetted_df = df[(df["score"] >= confidence_score) & (df["class_label"] == "tube_tip")]

    else:
        subsetted_df = df[(df["score"] >= confidence_score) &
                          (df["class_label"].isin(["germinated",
                                                   "ungerminated",
                                                   "burst",
                                                   "unknown_germinated",
                                                   "aborted"]))]

    return subsetted_df


def calculate_centroid(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the centroid value for each row of a dataframe.

    The dataframe should have columns titled "ymin", "xmin", "ymax", "xmax".

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns:
    -------
    df : pd.DataFrame
        Dataframe with added columns "centroid_x" and "centroid_y".

    """

    # Coordinates are scaled. Returning to the original dimensions here.
    original_x = 2048
    original_y = 2048

    df["centroid_x"] = df.apply(lambda row: (row["xmin"] + row["xmax"]) / 2 * original_x, axis=1)
    df["centroid_y"] = df.apply(lambda row: (row["ymin"] + row["ymax"]) / 2 * original_y, axis=1)

    return df


def add_rows_to_btrack(
    df: pd.DataFrame
) -> List[btrack.btypes.PyTrackObject]:
    """Create btrack PyTrackObjects from a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with calculated centroids and confidence score cutoffs.

    Returns
    -------
    btrack_objects: List[btrack.btypes.PyTrackObject]
        List of PyTrackObjects, one for each row of the input DataFrame.

    """

    btrack_objects = []
    id_counter = 0

    print(df)

    for row in df.itertuples(index=False):
        row_dict = {'ID': id_counter,
                    't': getattr(row, "timepoint"),
                    'x': getattr(row, "centroid_x"),
                    'y': getattr(row, "centroid_y"),
                    'z': 0.}

        obj = btrack.btypes.PyTrackObject.from_dict(row_dict)
        btrack_objects.append(obj)
        id_counter += 1

    return btrack_objects


def run_tracking(
    btrack_objects: Union[List[btrack.btypes.PyTrackObject], np.ndarray]
) -> Tuple[np.ndarray, dict, dict]:
    """Run the btrack tracking algorithm on a list of PyTrackObjects.

    Parameters
    ----------
    btrack_objects : Union[List[btrack.btypes.PyTrackObject], np.ndarray]
        A list of PyTrackObjects created with the add_rows_to_btrack function.

    Returns
    -------
    data : numpy.ndarray
        The track data in a format for napari visualization.
    properties : dict
        The properties of the tracks in a dictionary format.
    graph : dict
        The track graph in a dictionary format.

    """

    with btrack.BayesianTracker() as tracker:
        tracker.max_search_radius = 200
        tracker.tracking_updates = ["MOTION"]

        # configure the tracker using a config file
        # tracker.configure("./config/btrack_config.json")
        tracker.configure("./config/cell_config.json")

        # append the objects to be tracked
        tracker.append(btrack_objects)

        # set the tracking volume
        tracker.volume = ((0, 2048), (0, 2048))

        # track them (in interactive mode)
        tracker.track(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks in a format for napari visualization
        data, properties, graph = tracker.to_napari()

        return data, properties, graph


def visualize_tracks(
        track_data: np.ndarray,
        track_properties: dict,
        track_graph: dict,
        background_images: str,
        show_bounding_boxes: bool
) -> None:
    """Visualize btrack output with source images as background.

    Parameters
    ----------
    track_data : np.ndarray
        Output from the run_tracking function.
    track_properties : dict
        Output from the run_tracking function.
    track_graph : dict
        Output from the run_tracking function.
    background_images : str
        Name of the images used for the background.
    show_bounding_boxes : bool
        Whether or not to load the background images with bounding boxes.

    Returns
    -------
    None

    """

    print("Opening Napari viewer")
    viewer = napari.Viewer()

    print("Adding images")
    image_series = []

    # Formatting the image background image string
    if show_bounding_boxes:
        background_images = background_images + "_stab_inference"
    else:
        background_images = background_images + "_stab"

    # Getting the path of the background images
    current_script_path = Path(__file__).resolve()

    # Navigate to the parent directory
    parent_dir = current_script_path.parent.parent

    # Navigate to the data directory
    image_dir = parent_dir / "data" / "btrack_visualization_images" / background_images

    for image_path in sorted(image_dir.glob("*.jpg")):
        image = imread(image_path)
        image_array = np.asarray(image)
        image_series.append(image_array)
    viewer_array = np.asarray(image_series)
    viewer.add_image(viewer_array, scale=(1.0, 1.0, 1.0), name='images')

    print("Adding tracks")
    viewer.add_tracks(
        track_data,
        properties=track_properties,
        graph=track_graph,
        name="Tracks",
        tail_width=4,
        tail_length=1000,
        colormap="hsv",
        blending="Translucent",
        opacity=0.5,
        visible=True
    )

    napari.run()


def main():
    pd.set_option('display.max_columns', None)

    # Some example image sequence inference files
    # image_seq_name = "2022-03-03_run1_26C_C2"
    image_seq_name = "2022-03-07_run1_26C_B5"
    # image_seq_name = "2022-03-07_run1_26C_C2"

    # Loading and processing the dataframe
    df = load_tsv(image_seq_name)
    subsetted_df = subset_df(df, 0.35, "tip")
    subsetted_df = calculate_centroid(subsetted_df)

    # Adding to btrack and calculating tracks
    btrack_objects = add_rows_to_btrack(subsetted_df)
    data, properties, graph = run_tracking(btrack_objects)

    # Viewing tracks and images with Napari
    visualize_tracks(data, properties, graph, image_seq_name, show_bounding_boxes=True)

#     ####### EXPERIMENTAL #######
#
#     # Repeating for pollen classes
#     subsetted_df = subset_df(df, 0.35, "pollen")
#     subsetted_df = calculate_centroid(subsetted_df)
#
#     # Adding to btrack and calculating tracks
#     btrack_objects = add_rows_to_btrack(subsetted_df)
#     data, properties, graph = run_tracking(btrack_objects)
#
#     # Viewing tracks and images with Napari
#     visualize_tracks(data, properties, graph, image_seq_name, show_bounding_boxes=True)

    print("All done")


if __name__ == "__main__":
    main()
