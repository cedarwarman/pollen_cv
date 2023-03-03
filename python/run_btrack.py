#!/usr/bin/env python3
import btrack.btypes

# Trying out btrack for tracking pollen tubes.
# https://github.com/quantumjot/BayesianTracker

from typing import List, Tuple, Union
import pandas as pd
import numpy as np
from skimage.io import imread
from pathlib import Path
import glob
import btrack
from btrack.btypes import PyTrackObject
import napari


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

    return df


def subset_df(
    df: pd.DataFrame,
    confidence_score: float
) -> pd.DataFrame:
    """Subset a dataframe based on a confidence score and class.

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe.
    confidence_score : float
        Minimum score to filter the dataframe.

    Returns:
    -------
    subsetted_df : pd.DataFrame

    """
    subsetted_df = df[(df["score"] >= confidence_score) & (df["class"] == "tube_tip")]

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
) -> List[PyTrackObject]:
    """Create btrack PyTrackObjects from a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with calculated centroids and confidence score cutoffs.

    Returns
    -------
    btrack_objects: List[PyTrackObject]
        List of PyTrackObjects, one for each row of the input DataFrame.

    """

    btrack_objects = []
    id_counter = 0

    for row in df.itertuples(index=False):
        row_dict = {'ID': id_counter,
                    't': getattr(row, "timepoint"),
                    'x': getattr(row, "centroid_x"),
                    'y': getattr(row, "centroid_y"),
                    'z': 0.}

        # row_dict['rpn_score'] =
        # row_dict['rpn_obj_type'] =
        # row_dict['rpn_box'] =

        obj = PyTrackObject.from_dict(row_dict)
        btrack_objects.append(obj)
        id_counter += 1

    return btrack_objects


def run_tracking(
    btrack_objects: Union[List[PyTrackObject], np.ndarray]
) -> Tuple[np.ndarray, dict, dict]:
    """Run the btrack tracking algorithm on a list of PyTrackObjects.

    Parameters
    ----------
    btrack_objects : Union[List[PyTrackObject], np.ndarray]
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
        background_images: Union[str, Path]
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
    background_images : Union[str, Path]
        Path to the images used for the background.

    Returns
    -------
    None

    """

    print("Opening Napari viewer")
    viewer = napari.Viewer()

    print("Adding images")
    image_series = []

    for image_path in sorted(glob.glob(background_images)):
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
    df = load_tsv("2022-03-03_run1_26C_C2_t082_stab_predictions.tsv")
    # df = load_tsv("2022-03-07_run1_26C_B5_t082_stab_predictions.tsv")
    # df = load_tsv("2022-03-07_run1_26C_C2_t082_stab_predictions.tsv")

    print("DF is: ")
    print(df)

    df = subset_df(df, 0.35)
    df = calculate_centroid(df)
    print(df.head(n=5))
    btrack_objects = add_rows_to_btrack(df)
    # print(btrack_objects[0])
    data, properties, graph = run_tracking(btrack_objects)

    # Image sequences to go along with the inference, for visualization.
    image_dir = "/Users/warman/git/pollen_cv/data/btrack_visualization_images/2022-03-03_run1_26C_C2_inference/*.jpg"
    # image_dir = "/Users/warman/Desktop/Science/computer_vision/btrack/2022-03-03_run1_26C_C2_stab/*.jpg"

    # image_dir = "/Users/warman/Desktop/Science/computer_vision/btrack/2022-03-07_run1_26C_B5_inference/*.jpg"
    # image_dir = "/Users/warman/Desktop/Science/computer_vision/btrack/2022-03-07_run1_26C_B5_stab/*.jpg"

    # image_dir = "/Users/warman/Desktop/Science/computer_vision/btrack/2022-03-07_run1_26C_C2_inference/*.jpg"
    # image_dir = "/Users/warman/Desktop/Science/computer_vision/btrack/2022-03-07_run1_26C_C2_stab/*.jpg"

    # Viewing tracks and images with Napari
    visualize_tracks(data, properties, graph, image_dir)

    print("All done")


if __name__ == "__main__":
    main()
