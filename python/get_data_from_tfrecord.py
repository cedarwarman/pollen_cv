#!/usr/bin/env python3
"""Get data from tfrecord file.
This script pulls out image data and class counts from a tfrecord file. Class
counts are summarized per image.

Usage:
    python get_data_from_tfrecord.py \
        --tfrecord   \
        --output /media/volume/sdb/tfrecords/2023-04-13

Arguments:
    --tfrecord
        The tfrecord file to process.
    --output
        The save location.

"""

import argparse
import io
import os
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


def process_example(
    example: tf.train.Example
) -> Tuple[str, list, np.ndarray]:
    """Process a tfrecord example.
    Processes a TensorFlow Example object to extract the filename, class list, and image data.

    Parameters
    ----------
    example : tf.train.Example
        A TensorFlow Example object containing image data, class labels, and the filename.

    Returns
    -------
    Tuple[str, list, np.ndarray]
        A tuple containing the following elements:
            - filename (str): The image file name.
            - class_list (List[str]): A list of class labels associated with the image.
            - image_data (np.ndarray): A NumPy array containing the raw image data.
    """

    filename = ""
    class_list = []
    image_data = None
    for key, value in example.features.feature.items():
        if key == "image/filename":
            filename = value.bytes_list.value[0].decode('UTF-8')[:-9]
        elif key == "image/object/class/text":
            for value2 in value.bytes_list.value:
                class_string = value2.decode('UTF-8')
                class_list.append(class_string)
        elif key == "image/encoded":
            image_data = np.frombuffer(value.bytes_list.value[0], dtype=np.uint8)
    return filename, class_list, image_data


def save_data_frame(
    input_df: pd.DataFrame,
    output_path: str
) -> None:
    """Saves a Pandas data frame to a specified directory.

    Parameters
    ----------
    input_df : pd.DataFrame
        Pandas data frame to be saved.
    output_path : str
        Path where the data frame will be saved.

    Returns
    -------
    None
    """

    input_df.to_csv(pathlib.Path(output_path) / "ground_truth.tsv",
                    index=False,
                    sep='\t')


def save_image(
    filename: str,
    image_data: np.ndarray,
    save_dir: str
) -> None:
    """Saves the given image data as a jpg file in the specified directory.

    Parameters
    ----------
    filename : str
        The name of the image file without the file extension.
    image_data : np.ndarray
        A NumPy array containing the raw image data.
    save_dir : str
        The directory where the image will be saved.

    Returns
    -------
    None
    """

    image_buffer = io.BytesIO(image_data)
    img = Image.open(image_buffer).convert('L')
    save_path = os.path.join(save_dir, f"{filename}.jpg")
    img.save(save_path)


def parse_tfrecord(
    tfrecord_path: str,
    output: str
) -> None:
    """Parses a tfrecord file.

    Parameters
    ----------
    tfrecord_path : str
        Path to the tfrecord file.
    output : str
        Save path.

    Returns
    -------
    None

    """

    base_path = os.path.dirname(tfrecord_path)

    if not os.path.exists(output):
        os.makedirs(output)

    save_dir = os.path.join(base_path, "images")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    data = []

    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        filename, class_list, image_data = process_example(example)
        save_image(filename, image_data, output)
        for cls in class_list:
            data.append({"name": filename, "class": cls})
        print(filename, " Parsed")

    # Summarizing the counts
    combined_df = pd.DataFrame(data)
    combined_df = combined_df.groupby(["name", "class"], as_index=False).size().reset_index()

    save_data_frame(combined_df, output)


def main():
    parser = argparse.ArgumentParser(description='Gets images and class counts from a tfrecord file')
    parser.add_argument('--tfrecord', help="Path to tfrecord file")
    parser.add_argument('--output', help="Destination folder for the output")
    args = parser.parse_args()

    parse_tfrecord(args.tfrecord, args.output)


if __name__ == "__main__":
    main()

