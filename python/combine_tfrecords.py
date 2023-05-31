#!/usr/bin/env python3
"""Combine TFrecord files.
This script combines and shuffles two or more TFrecord files.

Usage:
    python combine_tfrecords.py \
            --records path/to/record_1.tfrecord path/to/record_2.tfrecord \
            --output path/to/output.tfrecord

Arguments:
    --records
        The paths to the TFrecords.
    --output
        The directory where the output will be saved.

"""

import argparse

import tensorflow as tf

def parse_arguments(
) -> argparse.Namespace:
    """Parses command line arguments.

    Returns
    -------
    parser : argparse.Namespace
        The parsed command line arguments.
    """

    parser = argparse.ArgumentParser(description='Combine and shuffle TFRecord files')

    parser.add_argument('--records', type=str, nargs='+',
                        help='One or more paths of the TFRecord files to be combined')

    parser.add_argument('--output', type=str, default="combined.tfrecord",
                        help='The filename of the output TFRecord file')

    return parser.parse_args()


def create_and_shuffle_dataset(
    filenames: List[str], 
    buffer_size: int = 10000
) -> tf.data.Dataset:
    """Creates a TFRecordDataset from the given filenames and shuffles it.

    Parameters
    ----------
    filenames : List[str]
        A list of filenames of the TFRecord files to be combined.
    buffer_size : int, optional
        The buffer size for shuffling. Default is 10000.

    Returns
    -------
    dataset : tf.data.Dataset
        The shuffled TFRecordDataset.
    """

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(buffer_size)

    return dataset


def write_dataset_to_file(
    dataset: tf.data.Dataset, 
    output_filename: str
) -> None:
    """Writes a dataset to a TFRecord file.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset to be written to a file.
    output_filename : str
        The filename of the output TFRecord file.
    """

    writer = tf.io.TFRecordWriter(output_filename)
    for serialized_example in dataset:
        writer.write(serialized_example.numpy())
    writer.close()


def main():
    # Parse arguments
    args = parse_arguments()

    # Create and shuffle dataset
    dataset = create_and_shuffle_dataset(args.filenames)

    # Write dataset to file
    write_dataset_to_file(dataset, args.output)


if __name__ == "__main__":
    main()
