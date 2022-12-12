#!/usr/bin/env python3

# Gets image names and class counts from a tfrecord file

import tensorflow as tf
import pandas as pd
import argparse
import pathlib

# Arguments
parser = argparse.ArgumentParser(description='Gets image names and class counts from a tfrecord file')
parser.add_argument('--tfrecord', help="Path to tfrecord file")
parser.add_argument('--output', help="Destination folder for the output")
args = parser.parse_args()

def parse_tfrecord(tfrecord_path):
    """Parses a tfrecord file

    Args:
        tfrecord_path: path to the tfrecord file

    Returns:
        Pandas data frame with object count summaries for each image
    """
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    combined_df = pd.DataFrame()

    for raw_record in raw_dataset:
        class_list = []
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        for key, value in example.features.feature.items():
            if key == "image/filename":
                filename = value.bytes_list.value[0].decode('UTF-8')[:-9]
            if key == "image/object/class/text":
                for value2 in value.bytes_list.value:
                    class_string = value2.decode('UTF-8')
                    class_list.append(class_string)

        print("Parsing", filename)
        df = pd.DataFrame()
        df["class"] = class_list
        df["name"] = filename
        combined_df = pd.concat([combined_df, df])

    # Summarizing the counts
    combined_df = combined_df.groupby(["name", "class"], as_index = False)

    return combined_df.size()

def save_data_frame(input_df, output_path):
    """Saves a data frame

    Takes in a Pandas data frame and saves it to a directory. 

    Args:
        input_df: Pandas data frame to be saved
        output_path: path where the data frame will be saved

    Returns:
        Nothing
    """

    input_df.to_csv(pathlib.Path(output_path) / "ground_truth.tsv",
            index = False,
            sep = '\t')

def main():
    parsed_tfrecord = parse_tfrecord(args.tfrecord)
    save_data_frame(parsed_tfrecord, args.output)

if __name__ == "__main__":
    main()

