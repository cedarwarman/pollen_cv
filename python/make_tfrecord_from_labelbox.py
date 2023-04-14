#!/usr/bin/env python3
# Testing remote development

# Based off of https://github.com/caseydaly/LabelboxToTFRecord/

import yaml
import os
from datetime import datetime
import random
from pathlib import Path
import labelbox
import argparse
import tensorflow as tf
from collections import OrderedDict

from object_detection.utils import dataset_util
from object_detection.protos import string_int_label_map_pb2
from google.protobuf import text_format

from typing import Dict, List, Tuple


class Label:
    """A class representing annotations.
    Includes bounding boxes and labels.

    Attributes
    ----------
    xmin : int
        The minimum x-coordinate of the bounding box.
    xmax : int
        The maximum x-coordinate of the bounding box.
    ymin : int
        The minimum y-coordinate of the bounding box.
    ymax : int
        The maximum y-coordinate of the bounding box.
    label : int or str
        The class id.

    Methods
    -------
    __repr__()
        Returns a string representation of the Label object.

    """

    def __init__(self, xmin, xmax, ymin, ymax, label):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.label = label

    def __repr__(self):
        return "Label({0}, {1}, {2}, {3}, {4}, {5})".format(self.xmin, self.xmax, self.ymin, self.ymax, self.label)


class TFRecordInfo:
    """A class to store multiple Label classes
    Stores multiple Label classes and some metadata. Becomes the tfrecord.

    Attributes
    ----------
    height : int
        The height of the image.
    width : int
        The width of the image.
    filename : str
        The filename of the image.
    encoded : bytes
        The encoded image data.
    format : str
        The image format (e.g., 'jpg').
    labels : list
        A list of label objects associated with the image.

    Methods
    -------
    __repr__():
        Returns a string representation of the TFRecordInfo object.

    """

    def __init__(self, height, width, filename, encoded, format, labels):
        self.height = height
        self.width = width
        self.filename = filename
        self.encoded = encoded
        self.format = format
        self.labels = labels

    def __repr__(self):
        return (
            "TFRecordInfo({0}, {1}, {2}, {3}, {4}, {5})"
            .format(self.height, self.width, self.filename,
                    type(self.encoded), self.format, self.labels)
        )


def open_yaml(
    yaml_path: Path
) -> dict:
    """Open Labelbox secrets.
    Open Labelbox secrets YAML file and load it as a dictionary.

    Parameters
    ----------
    yaml_path : str
        The path to the YAML file.

    Returns
    -------
    yaml_dict : dict
        The dictionary containing the YAML data.

    """

    with open(yaml_path) as file:
        yaml_dict = yaml.safe_load(file)
    return yaml_dict


def download_labels(
    api_key: dict,
    project_id: str
) -> dict:
    """Download the labels from a Labelbox project.

    Parameters
    ----------
    api_key : str
        The Labelbox API key.
    project_id : str
        The ID of the Labelbox project.

    Returns
    -------
    labels : dict
        The labels downloaded from the Labelbox project.

    """

    # Create Labelbox client
    lb = labelbox.Client(api_key=api_key)

    project = lb.get_project(project_id)

    # Export labels created in the selected date range as a json file:
    labels = project.export_labels(download=True)

    return labels


def label_from_labelbox_obj(
    lbl_box_obj: dict
) -> Label:
    """Create a Label object from a Labelbox label.

    Parameters
    ----------
    lbl_box_obj : dict
        A list containing annotations from Labelbox.

    Returns
    -------
    Label
        A Label object with bounding box coordinates and label identifier.

    """

    ymin = lbl_box_obj["bbox"]["top"]
    xmin = lbl_box_obj["bbox"]["left"]
    ymax = ymin + lbl_box_obj["bbox"]["height"]
    xmax = xmin + lbl_box_obj["bbox"]["width"]

    return Label(xmin, xmax, ymin, ymax, lbl_box_obj["value"])


def tube_tip_label_from_labelbox_obj(
    lbl_box_obj: dict,
    image_date: datetime
) -> Label:
    """Generate a tube tip label object from a labelbox object.

    Parameters
    ----------
    lbl_box_obj : dict
        A dictionary containing the labelbox object with keys "point" and "value".
        The "point" key contains a dictionary with keys "x" and "y" representing
        the x and y coordinates of the point.
    image_date : datetime
        The date the image was captured.

    Returns
    -------
    Label
        A Label object representing the tube tip label with xmin, xmax, ymin,
        ymax, and class id.

    """

    # Image dimensions are different depending on the date
    camera_switch_date = datetime(2022, 5, 27)
    image_w = 2048 if image_date <= camera_switch_date else 1600
    image_h = 2048 if image_date <= camera_switch_date else 1200

    # Size of bounding box around point
    box_edge_len = 35 if image_date <= camera_switch_date else 18

    point_x = lbl_box_obj["point"]["x"]
    point_y = lbl_box_obj["point"]["y"]

    # Makes sure bounding box doesn't go off the edge of the image
    xmin = max(0, point_x - 0.5 * box_edge_len)
    xmax = min(image_w, point_x + 0.5 * box_edge_len)
    ymin = max(0, point_y - 0.5 * box_edge_len)
    ymax = min(image_h, point_y + 0.5 * box_edge_len)

    # If the tube tip label is "tube_tip_bulging" then change it to "tube_tip"
    if lbl_box_obj["value"] == "tube_tip_bulging":
        lbl_box_obj["value"] = "tube_tip"

    # Ignore tube_tip_burst
    if lbl_box_obj["value"] != "tube_tip_burst":
        return Label(xmin, xmax, ymin, ymax, lbl_box_obj["value"])


def parse_labelbox_data(
    data: List[Dict],
    label_arg: str
) -> List[TFRecordInfo]:
    """Convert Labelbox data to a list of TFRecordInfo objects.

    Parameters
    ----------
    data : List[Dict]
        A list of dictionaries containing Labelbox data for each image.
    label_arg : str
        A string specifying which labels to include in the output.
        Options: "all", "pollen", "tube_tip".

    Returns
    -------
    records : List[TFRecordInfo]
        A list of TFRecordInfo objects.

    """

    records = []
    image_format = b'jpg'

    for record in data:
        image_name = record['External ID']
        image_date = datetime.strptime(image_name.split("_")[0], "%Y-%m-%d")

        print(f'Importing {image_name}')
        image_path = path_from_filename(image_name)

        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_jpg = fid.read()

        labels = []
        label_objs = record["Label"]["objects"]

        # Checks to see if the user wants all the labels, just the pollen
        # labels, or just the tube tip labels
        if label_arg == "all":
            label_list = label_objs
        elif label_arg == "pollen":
            label_list = [l for l in label_objs if l.get("bbox")]
        else:  # label_arg == "tube_tip"
            label_list = [l for l in label_objs if not l.get("bbox")]

        # If necessary, converts tube tip points to bounding boxes
        for l in label_list:
            if l.get("bbox"):
                labels.append(label_from_labelbox_obj(l))
            else:
                append_item = tube_tip_label_from_labelbox_obj(l, image_date)
                if append_item is not None:
                    labels.append(append_item)

        width, height = (2048, 2048) if image_date <= datetime(2022, 5, 27) else (1600, 1200)

        records.append(TFRecordInfo(width, height, image_name, encoded_jpg, image_format, labels))

    print(f"{len(data)} labeled images parsed")
    
    return records


def get_classes_from_labelbox(
    data: List[Dict],
    label_arg: str
) -> Dict[str, int]:
    """Extract unique class labels from labelbox data.

    Parameters
    ----------
    data : List[Dict]
        A list of dictionaries containing labelbox data.
    label_arg : str
        A string specifying which labels to include in the output.
        Options: "all", "pollen", "tube_tip".

    Returns
    -------
    labels : Dict[str, int]
        A dictionary with class labels as keys and integer indices as values.

    """

    labels_set = set()
    for record in data:
        if isinstance(record, dict) and "objects" in record["Label"]:
            label_objs = record["Label"]["objects"]

            for obj in label_objs:
                # These are removed or changed to tube_tip in the function
                # "tube_tip_label_from_labelbox_obj".
                if obj["value"] in {"tube_tip_bulging", "tube_tip_burst"}:
                    continue

                if label_arg == "all":
                    labels_set.add(obj["value"])
                elif label_arg == "pollen" and obj.get("bbox"):
                    labels_set.add(obj["value"])
                elif label_arg == "tube_tip" and not obj.get("bbox"):  # Tube tip classes
                    labels_set.add(obj["value"])

    labels_list = sorted(list(labels_set))
    labels = {label: idx for idx, label in enumerate(labels_list)}

    return labels


def validate_splits(
    args_splits: List[int],
    parser: argparse.ArgumentParser
) -> List[int]:
    """Validate input splits and adjust if necessary.

    Parameters
    ----------
    args_splits : List[int]
        A list of integer values representing dataset split percentages.
    parser : argparse.ArgumentParser
        The argparse instance used for error handling.

    Returns
    -------
    List[int]
        A list of adjusted and sorted integer values representing dataset split percentages.

    """

    if not args_splits:
        return [100]

    if any(s <= 0 for s in args_splits):
        parser.error(message='splits must be positive integers')

    splits_sum = sum(args_splits)
    if splits_sum < 100:
        args_splits.append(100 - splits_sum)
    elif splits_sum > 100:
        parser.error("splits must sum up to <= 100")

    return sorted(args_splits)


def splits_to_record_indices(
    splits: List[int],
    num_records: int
) -> List[int]:
    """Convert percentage splits to record indices based on the total number of records.

    Parameters
    ----------
    splits : List[int]
        A list of integer values representing dataset split percentages.
    num_records : int
        The total number of records in the dataset.

    Returns
    -------
    List[int]
        A list of integer values representing the indices corresponding to the dataset splits.

    """

    if not splits or splits == [100]:
        return [num_records]

    if sum(splits) != 100:
        raise ValueError("Percentages must add to 100")

    prev_idx = 0
    img_indices = []

    for split in splits[:-1]:
        end_img_idx = round(prev_idx + (split / 100) * num_records)
        if end_img_idx != prev_idx:
            img_indices.append(min(end_img_idx, num_records))
        prev_idx = end_img_idx

    img_indices.append(num_records)

    return list(OrderedDict.fromkeys(img_indices))


def path_from_filename(
    filename: str
) -> Path:
    """Get the local image file path.
    This function is not generalizable, it uses specific paths where images
    from different dates were located. It will need to be modified if there
    is any change in file naming format or local directory structure.

    Parameters
    ----------
    filename : str
        The filename of the image.

    Returns
    -------
    out_path: Path
        The local path to the image file.

    """

    split_string = filename.split("_")
    image_date = datetime.strptime(split_string[0], "%Y-%m-%d")

    if image_date <= datetime(2022, 5, 27):
        dir_string = f"{split_string[0]}_{split_string[1]}_{split_string[2]}_stab"
        path_base = Path('/media/volume/sdb/jpgs')
    else:
        dir_string = f"{split_string[0]}_{split_string[1]}_{split_string[2]}_normalized_stabilized"
        path_base = Path('/media/volume/sdb/norm_stab_jpgs')

    out_path = path_base / dir_string / f'well_{split_string[3]}' / filename
    
    return out_path


def create_tf_example(record_obj, class_dict):
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for label_obj in record_obj.labels:
        xmins.append(label_obj.xmin / record_obj.width)
        xmaxs.append(label_obj.xmax / record_obj.width)
        ymins.append(label_obj.ymin / record_obj.height)
        ymaxs.append(label_obj.ymax / record_obj.height)
        classes_text.append(label_obj.label.encode('utf8'))
        # print(f"Class_dict at label_obj.label with one added is: {class_dict[label_obj.label] + 1}")
        # To match the classes in class_dict
        classes.append(class_dict[label_obj.label] + 1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(record_obj.height),
        'image/width': dataset_util.int64_feature(record_obj.width),
        'image/filename': dataset_util.bytes_feature(record_obj.filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature("empty".encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(record_obj.encoded),
        'image/format': dataset_util.bytes_feature(record_obj.format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# Maps a { 'shark': 1, 'person': 2 } dict to a labelmap structure
# for the pbtxt file
def class_dict_to_label_map_str(class_dict):
    label_map_proto = string_int_label_map_pb2.StringIntLabelMap()
    for key,val in class_dict.items():
        item = label_map_proto.item.add()
        item.name = key
        # 0 is reserved for 'background' only, which we aren't using
        item.id = val + 1

    return text_format.MessageToString(label_map_proto)

# Making the tfrecords with split
def generate_tfrecords(image_source, tfrecord_dest, splits, data, records, class_dict):
    print("Making tfrecords")

    # Hopefully from here it's pretty straightforward? Just need to add the 
    # images as an arg and deal with their paths in parse_labelbox_data and/or
    # create_tf_example, figure out what's actually necessary in create_tf_example,
    # and write it up.

    # Making a directory for the output tfrecords
    tfrecord_folder = tfrecord_dest
    # print(tfrecord_folder)
    if not os.path.exists(tfrecord_folder):
        os.makedirs(tfrecord_folder)
    # Might not need to do this if I'm using pathlib
    if tfrecord_folder[len(tfrecord_folder)-1] != '/':
        tfrecord_folder += '/'

    strnow = datetime.now().strftime('%Y-%m-%d_t%H%M%S')
    splits = splits_to_record_indices(splits, len(records))
    # print("splits is: ", splits)
    assert splits[-1] == len(records), f'{splits}, {len(records)}'

    random.shuffle(records)

    # This section ensures images with multiple records in the same time series all
    # end up in the training dataset to remove possible data leakage. If the training
    # dataset doesn't not have possible data leakage then this section can be removed.
    # It checks to see if images are part of a list of time series duplicates, and if 
    # so, moves them tot he end of the records list, which will put them in the 
    # largest split, which is the training set. If you are doing more than one split 
    # the behavior could be not what you expect (tested with -splits 80 20).
    image_duplicates = [
        "2021-12-20_run1_26C_A3_t048_stab.jpg",
        "2021-12-15_run1_34C_D4_t034_stab.jpg",
        "2022-01-25_run2_34C_C1_t007_stab.jpg",
        "2021-11-03_run1_34C_A3_t069_stab.jpg",
        "2022-03-18_run1_34C_D6_t050_stab.jpg",
        "2022-02-09_run1_26C_D2_t038_stab.jpg",
        "2021-12-02_run2_34C_C5_t073_stab.jpg",
        "2022-03-18_run1_34C_D6_t017_stab.jpg",
        "2022-03-21_run2_26C_C4_t005_stab.jpg",
        "2022-05-03_run2_34C_C4_t047_stab.jpg",
        "2022-03-21_run2_26C_C4_t041_stab.jpg",
        "2021-12-17_run2_34C_C2_t043_stab.jpg",
        "2021-12-02_run2_34C_C5_t074_stab.jpg",
        "2022-02-01_run1_34C_A5_t035_stab.jpg",
        "2022-04-28_run2_34C_D2_t036_stab.jpg",
        "2022-02-21_run1_26C_D4_t054_stab.jpg",
        "2022-03-09_run2_26C_C1_t069_stab.jpg",
        "2022-02-08_run1_34C_C6_t037_stab.jpg",
        "2022-04-14_run1_26C_D4_t073_stab.jpg",
        "2022-04-28_run2_34C_D2_t034_stab.jpg",
        "2022-02-08_run1_34C_C6_t077_stab.jpg",
        "2022-01-25_run2_34C_C1_t069_stab.jpg",
        "2022-02-09_run1_26C_D2_t003_stab.jpg",
        "2022-03-08_run1_34C_B3_t012_stab.jpg",
        "2021-12-17_run2_34C_D2_t076_stab.jpg",
        "2022-03-22_run1_34C_C3_t055_stab.jpg",
        "2021-11-19_run2_34C_A6_t063_stab.jpg",
        "2022-04-28_run2_34C_D2_t003_stab.jpg",
        "2022-05-09_run1_26C_D4_t027_stab.jpg",
        "2021-11-08_run1_34C_C1_t004_stab.jpg",
        "2022-02-21_run1_26C_D4_t082_stab.jpg",
        "2021-12-15_run1_34C_D4_t050_stab.jpg",
        "2022-04-21_run2_34C_C2_t056_stab.jpg",
        "2021-12-20_run1_26C_A3_t043_stab.jpg",
        "2022-03-09_run2_26C_C1_t078_stab.jpg",
        "2022-04-14_run1_26C_D4_t029_stab.jpg",
        "2022-04-21_run2_34C_C2_t064_stab.jpg",
        "2022-04-20_run2_26C_B6_t074_stab.jpg",
        "2021-11-08_run1_34C_C1_t059_stab.jpg",
        "2021-11-03_run1_34C_A3_t005_stab.jpg",
        "2022-03-22_run1_34C_C3_t067_stab.jpg",
        "2022-05-03_run2_34C_C4_t050_stab.jpg",
        "2022-03-08_run1_34C_B3_t004_stab.jpg",
        "2022-04-20_run2_26C_B6_t007_stab.jpg",
        "2021-12-17_run2_34C_D2_t062_stab.jpg",
        "2022-05-09_run1_26C_D4_t077_stab.jpg",
        "2021-11-19_run2_34C_A6_t060_stab.jpg",
        "2022-03-21_run1_26C_D4_t029_stab.jpg",
        "2022-02-01_run1_34C_A5_t038_stab.jpg",
        "2022-02-09_run2_26C_C1_t021_stab.jpg",
        "2022-03-21_run1_26C_D4_t009_stab.jpg",
        "2021-12-17_run2_34C_C2_t070_stab.jpg",
        "2022-02-09_run2_26C_C1_t010_stab.jpg"]
    record_index = 0
    #list_index = 0 # Just for printing during troubleshooting
    rearranged_records = records

    for record in records:
        if any(record.filename in x for x in image_duplicates):
            #list_index += 1
            #print(record.filename, "present in list, total =", list_index)

            # Moves it to the end of the records list
            rearranged_records.append(rearranged_records.pop(record_index))
        record_index += 1
    records = rearranged_records

    # Making the tfrecord files
    split_start = 0
    print(f'Creating {len(splits)} TFRecord files:')
    for split_end in splits:
        outfile = f'{strnow}_n{split_end - split_start}.tfrecord'
        # print(f"Outfile is: {outfile}")
        outpath = tfrecord_folder + outfile
        with tf.io.TFRecordWriter(outpath) as writer:
            for record in records[split_start:split_end]:
                print("Adding", record.filename, "to split ending in", split_end)
                tf_example = create_tf_example(record, class_dict)
                writer.write(tf_example.SerializeToString())
        print(f'Successfully created TFRecord file at: {outpath}')
        split_start = split_end

    pb_file_name = f'{strnow}.pbtxt'
    label_text = class_dict_to_label_map_str(class_dict)
    with open(tfrecord_folder + pb_file_name, 'w') as label_file:
        label_file.write(label_text)
        print(f'Successfully created label map file at: {tfrecord_folder + pb_file_name}')

def main():
    # Some args
    parser = argparse.ArgumentParser(description='Convert Labelbox data to TFRecord and store .tfrecord file(s) locally.')
    parser.add_argument('--tfrecord-dest', help="Destination folder for .tfrecord file(s)", default="tfrecord")
    parser.add_argument('--splits', help="Space-separated list of integer percentages for splitting the " +
        "output into multiple TFRecord files instead of one. Sum of values should be <=100. " +
        "Example: '--splits 10 70' will write 3 files with 10%%, 70%%, and 20%% of the data, respectively",
        nargs='+',
        type=int)
    parser.add_argument('--label_type',
        choices=["all", "pollen", "tube_tip"], help="Which labels to import",
        default="all")
    args = parser.parse_args()

    # Getting the secrets
    yaml_path = Path.home() / '.credentials' / 'labelbox.yaml'
    labelbox_secrets = open_yaml(yaml_path)
    api_key = labelbox_secrets["api_key"]
    project_id = labelbox_secrets["project_id"]

    # Downloading the labels
    labels = download_labels(api_key, project_id)

    # Parsing the labels
    records = parse_labelbox_data(labels, args.label_type)

    # Getting the different classes present
    class_dict = get_classes_from_labelbox(labels, args.label_type)

    splits = validate_splits(args.splits, parser)

    # Making the tfrecords
    generate_tfrecords("add_image_path", args.tfrecord_dest, splits, labels, records, class_dict)

if __name__ == "__main__":
    main()
