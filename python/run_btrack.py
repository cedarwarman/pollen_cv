#!/usr/bin/env python3

"""
Trying out btrack for tracking pollen tubes.

Original btrack repository is located here:
https://github.com/quantumjot/BayesianTracker
"""

from datetime import datetime
import os
from pathlib import Path
import argparse
from typing import List, Tuple, Union

import btrack
import btrack.btypes
import napari
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.neighbors import BallTree


def parse_arguments(
) -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns
    -------
    parser : argparse.Namespace
        Parsed arguments as a namespace object.

    Raises
    ------
    NotADirectoryError
        If a given directory path is not a directory.
    """

    def dir_path(string: str) -> str:
        if Path(string).exists():
            return string
        else:
            try:
                os.makedirs(string, exist_ok=True)
                return string
            except OSError as e:
                raise OSError(f"Error creating directory '{string}': {e}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference",
        type=str, help="Path to model inference file", required=True
    )
    parser.add_argument(
        "--images",
        type=dir_path,
        help="Path to directory containing images for visualization with Napari",
        required=False
    )
    parser.add_argument(
        "--output",
        type=dir_path,
        help="Path to directory where output will be saved",
        required=True
    )

    return parser.parse_args()


def load_tsv(
    file_path: str
) -> pd.DataFrame:
    """Load a tsv file into a pandas dataframe.

    Parameters:
    ----------
    file_path : str
        The path to the tsv file.

    Returns:
    -------
    df : pd.DataFrame
        The tsv file loaded into a pandas dataframe.

    """

    # Load the tsv file into a Pandas DataFrame
    df = pd.read_csv(file_path, sep="\t")

    # Fix a bad column name
    df = df.rename(columns={"class": "class_label"})

    return df


def get_image_dimensions(
    df: pd.DataFrame
) -> dict:
    """Get image dimensions from date.
    This experiment used two cameras which had different dimensions. The appropriate
    dimensions are found by looking at the date of the image sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    image_dimensions : dict
        x and y dimensions of the images.

    """

    image_date = pd.to_datetime(df["date"].iloc[0])
    camera_switch_date = datetime(2022, 5, 27)
    image_dimensions = {
        "x": 2048 if image_date <= camera_switch_date else 1600,
        "y": 2048 if image_date <= camera_switch_date else 1200,
    }

    return image_dimensions


def subset_df(
    df: pd.DataFrame,
    class_string: str
) -> pd.DataFrame:
    """Subset a dataframe based on a confidence score and class.
    Also removes the "unknown_germinated" class.

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe.
    class_string : str
        Class for subsetting the dataframe. Either "tip" or "pollen"

    Returns:
    -------
    subsetted_df : pd.DataFrame

    """

    # Subsetting by tube tip or pollen
    if class_string == "tip":
        df = df[df["class_label"] == "tube_tip"]
    else:
        # Removes "unknown_germinated"
        df = df[df["class_label"].isin(["germinated",
                                        "ungerminated",
                                        "burst",
                                        "aborted"])]

    # Subsetting by confidence score, based on the camera switch date.
    camera_switch_date = datetime(2022, 5, 27)

    subset_dict_camera_one = {"aborted": 0.33,
                              "ungerminated": 0.39,
                              "germinated": 0.36,
                              "burst": 0.23,
                              "tube_tip": 0.34,
                              "unknown_germinated": 0.25}
    subset_dict_camera_two = {"aborted": 0.29,
                              "ungerminated": 0.23,
                              "germinated": 0.35,
                              "burst": 0.16,
                              "tube_tip": 0.3,
                              "unknown_germinated": 0.34}


    def filter_rows(row):
        if datetime.strptime(row["date"], "%Y-%m-%d") <= camera_switch_date:
            return row["score"] >= subset_dict_camera_one[row["class_label"]]
        else:
            return row["score"] >= subset_dict_camera_two[row["class_label"]]

    mask = df.apply(filter_rows, axis=1)
    df = df[mask]

    return df


def calculate_centroid(
    df: pd.DataFrame,
    image_dimensions: dict
) -> pd.DataFrame:
    """Calculate the centroid value for each row of a dataframe.

    The dataframe should have columns titled "ymin", "xmin", "ymax", "xmax".

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe.
    image_dimensions : dict
        The x, y dimensions of the image.

    Returns:
    -------
    df : pd.DataFrame
        Dataframe with added columns "centroid_x" and "centroid_y".

    """

    # Coordinates are scaled. Returning to the original dimensions here.
    original_x = image_dimensions["x"]
    original_y = image_dimensions["y"]

    df = df.assign(
        centroid_x=(df["xmin"] + df["xmax"]) / 2 * original_x,
        centroid_y=(df["ymin"] + df["ymax"]) / 2 * original_y,
    )

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

    for row in df.itertuples(index=False):
        row_dict = {"ID": id_counter,
                    "t": getattr(row, "timepoint"),
                    "x": getattr(row, "centroid_x"),
                    "y": getattr(row, "centroid_y"),
                    "z": 0.,
                    "object_class": getattr(row, "class_label")}

        obj = btrack.btypes.PyTrackObject.from_dict(row_dict)
        btrack_objects.append(obj)
        id_counter += 1

    return btrack_objects


def run_tracking(
    btrack_objects: Union[List[btrack.btypes.PyTrackObject], np.ndarray],
    image_dimensions: dict
) -> Tuple[np.ndarray, dict, dict]:
    """Run the btrack tracking algorithm on a list of PyTrackObjects.

    Parameters
    ----------
    btrack_objects : Union[List[btrack.btypes.PyTrackObject], np.ndarray]
        A list of PyTrackObjects created with the add_rows_to_btrack function.
    image_dimensions : dict
        The x/y dimensions of the image

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
        # tracker.configure("../config/btrack/btrack_config.json")
        tracker.configure("../config/btrack/cell_config.json")

        # append the objects to be tracked
        tracker.append(btrack_objects)

        # Setting the tracking volume, depending on the camera the images have a
        # different size.
        tracker.volume = ((0, image_dimensions["x"]), (0, image_dimensions["y"]))

        # track them (in interactive mode)
        tracker.track(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks in a format for napari visualization. Adding
        # replace_na=False fixed a problem with adding class to PyTrackObjects.
        data, properties, graph = tracker.to_napari(replace_nan=False)

        # get the tracks in their native format (removed because redundant)
        # tracks = tracker.tracks

        return data, properties, graph


def visualize_tracks(
    track_data: np.ndarray,
    track_properties: dict,
    track_graph: dict,
    background_images: Union[str, Path],
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
        Name of the images used for the background.

    Returns
    -------
    None

    """

    viewer = napari.Viewer()
    image_series = []

    image_dir = Path(background_images)

    for image_path in sorted(image_dir.glob("*.jpg")):
        image = imread(image_path)
        image_array = np.asarray(image)
        image_series.append(image_array)
    viewer_array = np.asarray(image_series)
    viewer.add_image(viewer_array, scale=(1.0, 1.0, 1.0), name="images")

    viewer.add_tracks(
        track_data,
        properties=track_properties,
        graph=track_graph,
        name="Tracks",
        tail_width=4,
        tail_length=1000,
        colormap="hsv",
        blending="Opaque",
        # opacity=0.5,
        visible=True
    )

    napari.run()


def infer_pollen_classes(
    input_df: pd.DataFrame
) -> pd.DataFrame:
    """Add and correct pollen class information
    Infers missing classes from adjacent timepoints. Corrects class predictions that
    are biologically impossible. E.g., pollen must progress ungerminated > germinated
    > burst. It can start at any class (most often because a pollen grain floats down
    to the surface of the well), but cannot move backwards through the class
    progression.

    Parameters
    ----------
    input_df : pd.DataFrame

    Returns
    -------
    input_df : pd.DataFrame
        Dataframe with inferred classes.

    """
    # Making a copy of the original object_class column to make sure everything is
    # working as expected.
    input_df["original_object_class"] = input_df["object_class"].copy()

    # Replace NA with the value from the previous row
    input_df["object_class"].replace("nan", np.nan, inplace=True)
    input_df["object_class"].fillna(method="ffill", inplace=True)

    # If a pollen grain germinates or bursts then it can't be aborted and it must have
    # started as ungerminated. Here it's considered germinated if 3/4 classes are
    # germinated in a rolling window.
    def replace_func(group):
        # Create a boolean series where True if object_class is "germinated"
        is_germinated = group["object_class"] == "germinated"

        # Apply rolling window of size 4 and check if sum (number of "germinated") is >=3
        germinated_in_window = is_germinated.rolling(4).sum() >= 3

        # Replace every class before germination with ungerminated
        if germinated_in_window.any():
            first_germinated_index = is_germinated.idxmax()
            group.loc[:first_germinated_index, "object_class"] = "ungerminated"

        return group
    input_df.groupby("track_id", group_keys=False).apply(replace_func)

    # If the pollen grain class is aborted for >50% of the frames then it is
    # considered aborted for the entire track. Otherwise, aborted classes are
    # replaced with the previous class. First, calculating the percentage of "aborted"
    # for each track_id.
    total_counts = input_df.groupby("track_id").size()
    aborted_counts = input_df[input_df["object_class"] == "aborted"].groupby("track_id").size()
    percentage_aborted = aborted_counts / total_counts

    # Track IDs where "aborted" is more than 50%
    aborted_track_ids = percentage_aborted[percentage_aborted > 0.5].index

    # Change all the object_class to "aborted" for these track_ids
    input_df.loc[input_df["track_id"].isin(aborted_track_ids), "object_class"] = "aborted"

    # Track IDs where "aborted" is less than 50%
    not_aborted_track_ids = percentage_aborted[percentage_aborted <= 0.5].index

    # For these track_ids, replace "aborted" with the previous value, unless it's the
    # first one, then replace with "ungerminated".
    for track_id in not_aborted_track_ids:
        track_df = input_df.loc[input_df['track_id'] == track_id].copy()
        first_row_index = track_df.index[0]
        if track_df.loc[first_row_index, 'object_class'] == 'aborted':
            track_df.loc[first_row_index, 'object_class'] = 'ungerminated'
        track_df['object_class'] = track_df['object_class'].replace('aborted', method='ffill')
        input_df.loc[input_df['track_id'] == track_id, 'object_class'] = track_df['object_class']

    # For all other class conflicts, they must follow the ungerminated > germinated >
    # burst progression and class conflicts (switching from one to another not in the
    # progression) are settled by switching classes once 3/4 of consecutive classes
    # are the next class in the progression.
    progression = ["ungerminated", "germinated", "burst"]

    def process_group(group):
        group = group.copy()
        # Determine the starting class in the progression for this group
        start_class = group["object_class"].head(3).mode()[0]
        # Fixing a bug where ungerminated gets deleted if it's the first class
        if group.iloc[0]["object_class"] == "ungerminated":
            start_class = "ungerminated"

        # If it's aborted then they will all be aborted (see above) so will be
        # unaltered.
        if group["object_class"].iloc[0] == "aborted":
            return group
        start_index = progression.index(start_class)
        last_transition_index = group.index.min()
        # If the group never satisfies the requirements for a transition, we still want
        # to correct class errors, so we will deal with those groups at the end.
        made_transition = False

        # Apply the rules for each transition in the progression
        for i in range(start_index, len(progression) - 1):
            current_class = progression[i]
            next_class = progression[i + 1]
            is_next_class = group["object_class"] == next_class
            next_class_in_window = is_next_class.rolling(4).sum() >= 3

            # If there is a class transition, all the classes from the previous
            # transition up until the transition happens are the current class.
            if next_class_in_window.any():
                # If any window meets the condition, get the first index of this window
                first_next_class_index = group[is_next_class & next_class_in_window].index.min()
                group.loc[last_transition_index:first_next_class_index-3, "object_class"] = current_class
                last_transition_index = first_next_class_index
                made_transition = True
                if next_class == "burst":
                    # Reached the end of the progression, so everything after the
                    # transition is burst. But making sure it transitions in the right
                    # place and doesn't go back to germinated.
                    is_current_class = group["object_class"] == current_class
                    current_class_in_window = is_current_class.rolling(4).sum() == 4
                    # Checking to see if there are more of the current class after the
                    # first next_class transition.
                    has_true_after = current_class_in_window.loc[first_next_class_index-1:].any()
                    if has_true_after:
                        last_true_index = current_class_in_window.loc[first_next_class_index-1:][::-1].idxmax()
                        group.loc[last_transition_index-2:last_true_index, "object_class"] = "germinated"
                        group.loc[last_true_index + 1:, "object_class"] = "burst"

                    else:
                        group.loc[first_next_class_index-1:, "object_class"] = "burst"

            # If the transition never happens, all the remaining classes are the
            # current class, unless a class in the progression is skipped.
            else:
                if current_class == "ungerminated":
                    # Check to see if it goes straight to burst
                    is_burst = group["object_class"] == "burst"
                    burst_in_window = is_burst.rolling(4).sum() >= 3
                    if burst_in_window.any():
                        # Everything up until burst should be the current class.
                        first_burst_index = group[is_burst & burst_in_window].index.min()
                        group.loc[last_transition_index:first_burst_index - 3, "object_class"] = current_class
                        # Everything after burst should be burst.
                        group.loc[first_burst_index - 3:, "object_class"] = "burst"
                        break
                else:
                    group.loc[last_transition_index:, "object_class"] = current_class

        return group

    # Apply the function on each group
    input_df = input_df.groupby("track_id", group_keys=False).apply(process_group)

    # Now that the initial processing is done, do a screen for pollen tracks that are
    # not present in enough frames. Requiring 15 or more frames.
    input_df = input_df.groupby("track_id").filter(lambda x: len(x) >= 15)

    # Fill in classes when there's a gap.
    def fill_missing_rows(df):
        df.set_index(['track_id', 't'], inplace=True)
        df['filled_in_row'] = False

        filled_dfs = []
        for track_id in df.index.get_level_values(0).unique():
            reindexed_df = df.loc[track_id].reindex(
                range(df.loc[track_id].index.min(), df.loc[track_id].index.max() + 1))
            reindexed_df.index = pd.MultiIndex.from_product([[track_id], reindexed_df.index], names=['track_id', 't'])
            reindexed_df['filled_in_row'] = reindexed_df['filled_in_row'].isna()
            filled_dfs.append(reindexed_df)
        filled_df = pd.concat(filled_dfs)

        # Forward fill object_class after adding new rows
        filled_df['object_class'] = filled_df.groupby(level=0)['object_class'].ffill()

        # Interpolate x, y
        filled_df[['y', 'x']] = filled_df.groupby('track_id')[['y', 'x']].transform(
            lambda group: group.interpolate(method='linear'))

        filled_df.reset_index(inplace=True)

        return filled_df

    input_df = fill_missing_rows(input_df)

    return input_df


def infer_tube_tip_classes(
    input_df: pd.DataFrame
) -> pd.DataFrame:
    """Add and correct tube tip class information
    Corrects tube tip class information when missing.

    Parameters
    ----------
    input_df : pd.DataFrame

    Returns
    -------
    input_df : pd.DataFrame
        Dataframe with inferred classes.

    """
    # Making a copy of the original object_class column to make sure everything is
    # working as expected.
    input_df["original_object_class"] = input_df["object_class"].copy()

    # Replace nan with value from previous row.
    input_df["object_class"].replace("nan", np.nan, inplace=True)
    input_df["object_class"].fillna(method="ffill", inplace=True)

    # Screen out tube tip tracks that are present in less than 15 frames.
    input_df = input_df.groupby("track_id").filter(lambda x: len(x) >= 15)

    # Fill in classes when there's a gap.
    def fill_missing_rows(df):
        df.set_index(['track_id', 't'], inplace=True)
        df['filled_in_row'] = False

        filled_dfs = []
        for track_id in df.index.get_level_values(0).unique():
            reindexed_df = df.loc[track_id].reindex(
                range(df.loc[track_id].index.min(), df.loc[track_id].index.max() + 1))
            reindexed_df.index = pd.MultiIndex.from_product([[track_id], reindexed_df.index], names=['track_id', 't'])
            reindexed_df['filled_in_row'] = reindexed_df['filled_in_row'].isna()
            filled_dfs.append(reindexed_df)
        filled_df = pd.concat(filled_dfs)

        # Forward fill object_class after adding new rows
        filled_df['object_class'] = filled_df.groupby(level=0)['object_class'].ffill()

        # Interpolate x, y
        filled_df[['y', 'x']] = filled_df.groupby('track_id')[['y', 'x']].transform(
            lambda group: group.interpolate(method='linear'))

        filled_df.reset_index(inplace=True)

        return filled_df

    input_df = fill_missing_rows(input_df)

    # After filling in missing columns, if more than 50% of the original object classes
    # are nan, remove the track.
    input_df["original_object_class"].replace("nan", np.nan, inplace=True)
    nan_percentages = input_df.groupby("track_id")["original_object_class"].apply(
        lambda x: x.isna().mean()
    )
    bad_track_ids = nan_percentages[nan_percentages > 0.5].index
    input_df = input_df[~input_df["track_id"].isin(bad_track_ids)]

    return input_df


def make_pollen_df(
    track_data: np.ndarray,
    track_properties: dict
) -> pd.DataFrame:
    """Process pollen tracks for output in a data frame.

    Parameters
    ----------
    track_data : np.ndarray
        Output from the run_tracking function.
    track_properties : dict
        Output from the run_tracking function.

    Returns
    -------
    output_df : pd.DataFrame
        Dataframe that summarizes all the track and class information

    """
    properties_df = pd.DataFrame(track_properties)
    track_data_df = pd.DataFrame(
        track_data, columns=["root_track_data", "time", "y", "x"]
    )

    output_df = pd.concat([properties_df, track_data_df], axis=1)
    output_df = output_df.drop(
        ["state", "generation", "parent", "root_track_data", "time"], axis=1
    )
    output_df = output_df.rename(columns={"root": "track_id"})

    output_df = infer_pollen_classes(output_df)

    return output_df


def make_tube_tip_df(
    track_data: np.ndarray,
    track_properties: dict
) -> pd.DataFrame:
    """Process tube tip tracks for output in a data frame.

    Parameters
    ----------
    track_data : np.ndarray
        Output from the run_tracking function.
    track_properties : dict
        Output from the run_tracking function.

    Returns
    -------
    output_df : pd.DataFrame
        Dataframe that summarizes all the track and class information

    """
    properties_df = pd.DataFrame(track_properties)
    track_data_df = pd.DataFrame(
        track_data, columns=["root_track_data", "time", "y", "x"]
    )

    output_df = pd.concat([properties_df, track_data_df], axis=1)
    output_df = output_df.drop(
        ["state", "generation", "parent", "root_track_data", "time"], axis=1
    )
    output_df = output_df.rename(columns={"root": "track_id"})

    output_df = infer_tube_tip_classes(output_df)

    return output_df


def link_tubes_to_pollen(
    pollen_df: pd.DataFrame,
    tube_df: pd.DataFrame
) -> pd.DataFrame:
    """Link tube tracks to pollen tracks.
    Finds closest tracks with Euclidean distance, links tracks.

    Parameters
    ----------
    pollen_df : pd.DataFrame
        Processed pollen tracks.
    tube_df : pd.DataFrame
        Processed tube tip tracks.

    Returns
    -------
    ADD EVERYTHING ELSE YOU NEED FOR NAPARI
    output_df : pd.DataFrame
        Data frame with tube tip tracks linked to pollen tracks.

    """

    # First, crop dfs and reset track ids so that they are all unique. This allows a
    # parent / child relationship to be encoded in a graph as input to Napari for
    # visualization.
    pollen_df = pollen_df.iloc[:, :5]
    tube_df = tube_df.iloc[:, :5]
    pollen_df["track_id"] = pd.factorize(pollen_df["track_id"])[0]
    tube_df["track_id"] = (
        pd.factorize(tube_df["track_id"])[0] + pollen_df["track_id"].max() + 1
    )

    # Next setting up the nearest neighbors algorithm. Using BallTrees, but the sklearn
    # implementation will switch to brute force if there's less than 40 objects. Start
    # by making a cropped version of the pollen df so that tubes cannot be linked to
    # aborted or burst pollen.
    pollen_df_filtered = pollen_df[
        pollen_df.object_class.isin(["germinated", "ungerminated"])
    ]
    # Creating a dictionary of BallTrees for each unique time in pollen_df.
    trees = {
        time: BallTree(pollen_df_filtered.loc[pollen_df_filtered.t == time][["x", "y"]])
        for time in pollen_df_filtered.t.unique()
    }

    # Create a dictionary to hold the corresponding track_ids
    id_dict = {
        time: pollen_df_filtered.loc[pollen_df_filtered.t == time]["track_id"].values
        for time in pollen_df_filtered.t.unique()
    }

    # Function to find the nearest point in the pollen_df for a given point in the
    # tube_df at the same time.
    def get_nearest_track(row):
        point = [[row.x, row.y]]
        time = row.t
        if time in trees:
            dist, ind = trees[time].query(point, k=1)
            return id_dict[time][ind[0][0]]
        else:
            return None
    # Add a new column to the tube_df with the id of the nearest track point in the
    # pollen_df at the same time.
    tube_df["closest_pollen_track"] = tube_df.apply(get_nearest_track, axis=1)

    # Gets the closest pollen grain over time, so this replaces all of them with the
    # one that it was closest to at the start.
    most_common_in_first_three = tube_df.groupby("track_id").apply(
        lambda group: group.head(3)["closest_pollen_track"].mode().iat[0]
    )
    tube_df["closest_pollen_track"] = tube_df["track_id"].map(most_common_in_first_three)

    # Making outputs for visualization
    output_df = pd.concat([pollen_df, tube_df]).reset_index(drop=True)
    # Adding parent information. If a track is a pollen grain then the parent is
    # itself.
    output_df["closest_pollen_track"] = output_df["closest_pollen_track"].fillna(
        output_df["track_id"]
    )
    data_array = output_df.iloc[:, [0,1,3,4]].to_numpy(dtype="float64")
    properties_dict = {
        "t": output_df["t"].to_numpy(dtype='int64'),
        "state": np.full(len(output_df), 5, dtype="int64"),
        "generation": np.full(len(output_df), 0, dtype="int64"),
        "root": output_df["track_id"].to_numpy(dtype="int64"),
        "parent": output_df["closest_pollen_track"].to_numpy(dtype="int64"),
        "object_class": output_df["object_class"].to_numpy(dtype="<U32"),
    }

    # Going to try to encode the parent / child relatinoships in this graph dict, which
    # is an input for Napari.
    graph_dict = {}

    return output_df, data_array, properties_dict, graph_dict


def save_df_as_tsv(
    df: pd.DataFrame,
    linked_df: pd.DataFrame,
    output_path: Union[str, Path]
) -> None:
    """Save pandas data frame as a tsv file.
    Also adds metadata about the run.

    Parameters
    ----------
    df : pd.DataFrame
        Original data frame to pull some metadata from.
    linked_df : pd.DataFrame
        DataFrame to be saved as tsv.
    output_path : Union[str, Path]
        The path where the tsv file will be saved.

    Returns
    -------
    None

    """
    # Getting the run metadata from the original data frame and adding it to the
    # output data frame.
    metadata_row = df.loc[0, ["date", "run", "well", "tempc"]]
    metadata_df = pd.DataFrame([metadata_row] * len(linked_df)).reset_index(drop=True)
    linked_df = pd.concat([metadata_df, linked_df], axis=1)

    # Making the output file string
    file_name = (
            str(metadata_row["date"])
            + "_run"
            + str(metadata_row["run"])
            + "_"
            + str(metadata_row["tempc"])
            + "C_"
            + str(metadata_row["well"])
            + "_tracks.tsv"
    )

    # Getting the name of the image sequence from the df
    save_path = Path(output_path) / file_name

    linked_df.to_csv(save_path, sep='\t', index=False)


def main():
    print("Parsing args")
    args = parse_arguments()

    print("Loading data")
    df = load_tsv(args.inference)

    print("Calculating centroids")
    image_dimensions = get_image_dimensions(df)
    df = calculate_centroid(df, image_dimensions)

    ### POLLEN ###
    print("Pollen --- Subsetting data")
    subsetted_df = subset_df(df, "pollen")

    print("Pollen --- Calculating tracks")
    btrack_objects = add_rows_to_btrack(subsetted_df)
    data, properties, graph = run_tracking(btrack_objects, image_dimensions)

    print("Pollen --- Making dataframe")
    pollen_track_df = make_pollen_df(data, properties)

    ### TUBE TIPS ###
    print("Tube tip --- Subsetting data")
    subsetted_df = subset_df(df, "tip")

    print("Tube tip --- Calculating tracks")
    btrack_objects = add_rows_to_btrack(subsetted_df)
    data, properties, graph = run_tracking(btrack_objects, image_dimensions)

    print("Tube tip --- Making dataframe")
    tube_tip_track_df = make_tube_tip_df(data, properties)

    ### LINKING TUBES TO POLLEN
    print("Linking tubes to pollen")
    linked_df, data_array, properties_dict, graph_dict = link_tubes_to_pollen(
        pollen_track_df, tube_tip_track_df
    )

    print("Saving data frame")
    save_df_as_tsv(df, linked_df, args.output)

    print("Visualizing")
#    visualize_tracks(data, properties, graph, image_seq_name)
    visualize_tracks(data_array, properties_dict, graph_dict, args.images)

    print("All done")


if __name__ == "__main__":
    main()
