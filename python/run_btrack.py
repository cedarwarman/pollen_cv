#!/usr/bin/env python3

"""
Trying out btrack for tracking pollen tubes.

Original btrack repository is located here:
https://github.com/quantumjot/BayesianTracker
"""

from datetime import datetime
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
    file_name = file_name + "_predictions.tsv"

    # Get the path of the current script
    current_script_path = Path(__file__).resolve()

    # Navigate to the parent directory
    parent_dir = current_script_path.parent.parent

    # Navigate to the data directory
    data_dir = parent_dir / "data" / "cv_model_inference" / "predictions"

    # Get the path of the tsv file
    tsv_file = data_dir / file_name

    # Load the tsv file into a Pandas DataFrame
    df = pd.read_csv(tsv_file, sep="\t")

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
        background_images: str,
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

    Returns
    -------
    None

    """

    viewer = napari.Viewer()
    image_series = []
    background_images = background_images + "_inference"

    # Getting the path of the background images
    current_script_path = Path(__file__).resolve()

    # Navigate to the parent directory
    parent_dir = current_script_path.parent.parent

    # Navigate to the data directory
    image_dir = parent_dir / "data" / "cv_model_inference" / "images" / background_images

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


def infer_classes(
    input_df: pd.DataFrame
) -> pd.DataFrame:
    """Add and correct class information
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
        start_class = group['object_class'].head(3).mode()[0]
        # If it's aborted then they will all be aborted (see above) so will be
        # unaltered.
        if group["object_class"].iloc[0] == "aborted":
            return group
        start_index = progression.index(start_class)
        last_transition_index = group.index.min()
        # If the group never satisfies the requirements for a transition, we still want
        # to correct class errors, so we will deal with those groups at the end.
        made_transition = False
        print("\nHead")
        print(group.head(1))

        # Apply the rules for each transition in the progression
        for i in range(start_index, len(progression) - 1):
            current_class = progression[i]
            print("Current class: ", current_class)

            next_class = progression[i + 1]
            print("Next class: ", next_class)

            is_next_class = group["object_class"] == next_class
            next_class_in_window = is_next_class.rolling(4).sum() >= 3

            # If there is a class transition, all the classes from the previous
            # transition up until the transition happens are the current class.
            if next_class_in_window.any():
                print("Found class in a window, replacing up to the transition with current class")
                # If any window meets the condition, get the first index of this window
                first_next_class_index = group[is_next_class & next_class_in_window].index.min()
                group.loc[last_transition_index:first_next_class_index-3, "object_class"] = current_class
                last_transition_index = first_next_class_index
                made_transition = True
                if next_class == "burst":
                    # Reached the end of the progression, so everything after the
                    # transition is burst. But making sure it transitions in the right
                    # place and doesn't go back to germinated.
                    ###################################################################
                    is_current_class = group["object_class"] == current_class
                    current_class_in_window = is_current_class.rolling(4).sum() == 4
                    # Checking to see if there are more of the current class after the
                    # first next_class transition.
                    has_true_after = current_class_in_window.loc[first_next_class_index-1:].any()
                    if has_true_after:
                        print("has_true_after")
                        last_true_index = current_class_in_window.loc[first_next_class_index-1:][::-1].idxmax()
                        group.loc[last_transition_index-2:last_true_index, "object_class"] = "germinated"
                        print("Last true index: ", last_true_index)
                        group.loc[last_true_index + 1:, "object_class"] = "burst"

                    else:
                        group.loc[first_next_class_index-1:, "object_class"] = "burst"
            # If the transition never happens, all the remaining classes are the
            # current class, unless a class in the progression is skipped.
            else:
                if current_class == "ungerminated":
                    print("Current class is ungerminated")
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
                    print("Replacing everthing from ", last_transition_index, "with ", current_class)
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


def make_output_df(
    track_data: np.ndarray,
    track_properties: dict
) -> pd.DataFrame:
    """Visualize btrack output with source images as background.

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

    output_df = infer_classes(output_df)

    return output_df


def main():
    # Some example image sequence inference files
    image_seq_name = "2022-01-05_run1_26C_D2"
    # image_seq_name = "2022-03-07_run1_26C_B5"
    # image_seq_name = "2022-03-07_run1_26C_C2"
    # image_seq_name = "2022-06-05_run1_34C_A6"
    # image_seq_name = "2022-06-05_run1_34C_B1"
    # image_seq_name = "2022-06-05_run1_34C_B3"

    print("Loading data")
    df = load_tsv(image_seq_name)

    # Subsetting with pollen classes and confidence score threhsolds, removing
    # unknown_germinated class.
    print("Subsetting data")
    subsetted_df = subset_df(df, "pollen")

    print("Calculating centroids")
    image_dimensions = get_image_dimensions(df)
    subsetted_df = calculate_centroid(subsetted_df, image_dimensions)

    # Adding to btrack and calculating tracks
    print("Calculating tracks")
    btrack_objects = add_rows_to_btrack(subsetted_df)
    data, properties, graph = run_tracking(btrack_objects, image_dimensions)

    # Making a dataframe with all the track and class info
    print("Making dataframe")
    track_df = make_output_df(data, properties)

    # Viewing tracks and images with Napari
    print("Visualizing")
    visualize_tracks(data, properties, graph, image_seq_name)

    print("All done")


if __name__ == "__main__":
    main()
