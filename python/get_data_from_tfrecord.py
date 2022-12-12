#!/usr/bin/env python3

# Gets image names and class counts from a tfrecord file

import tensorflow as tf
import argparse

# Arguments
parser = argparse.ArgumentParser(description='Gets image names and class counts from a tfrecord file')
parser.add_argument('--tfrecord', help="Path to tfrecord file")
parser.add_argument('--output', help="Destination folder for the output")
args = parser.parse_args()

def open_tfrecord(tfrecord_path):
    """Open a tfrecord file

    Args:
        tfrecord_path: path to the tfrecord file

    Returns:
    """
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        for key, value in example.features.feature.items():
            if key == "image/filename":
                filename = value
                print(filename)
                for value2 in filename.bytes_list.value:
                    print(value2.decode('UTF-8'))
            if key == "image/object/class/text":
                print(value)
                for value2 in value.bytes_list.value:
                    print(value2.decode('UTF-8'))




def main():
    open_tfrecord(args.tfrecord)



if __name__ == "__main__":
    main()
