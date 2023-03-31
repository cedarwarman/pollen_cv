#!/usr/bin/env python3
# Testing remote development

# Largely based off of https://github.com/caseydaly/LabelboxToTFRecord/

import yaml
import io
import os
from datetime import datetime
import random
from pathlib import Path
import labelbox
import argparse
import tensorflow as tf
from PIL import Image
from collections import OrderedDict

from object_detection.utils import dataset_util
from object_detection.protos import string_int_label_map_pb2
from google.protobuf import text_format

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


# Downloading the labels
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

    # Get project by ID
    project = lb.get_project(project_id)

    # Export labels created in the selected date range as a json file:
    labels = project.export_labels(download=True)

    return labels


# Making a "label" class to store annotations and info for a single image
class Label:

    def __init__(self, xmin, xmax, ymin, ymax, label, text=""):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.label = label
        self.text = text

    def __repr__(self):
        return "Label({0}, {1}, {2}, {3}, {4}, {5})".format(self.xmin, self.xmax, self.ymin, self.ymax, self.label, self.text)


def label_from_labelbox_obj(lbl_box_obj):
    ymin = lbl_box_obj["bbox"]["top"]
    xmin = lbl_box_obj["bbox"]["left"]
    ymax = ymin + lbl_box_obj["bbox"]["height"]
    xmax = xmin + lbl_box_obj["bbox"]["width"]
    return Label(xmin, xmax, ymin, ymax, lbl_box_obj["value"])

def tube_tip_label_from_labelbox_obj(lbl_box_obj, image_date):
    # Camera switch date
    camera_switch_date = datetime(2022, 5, 27)

    # Image dimensions
    image_w = 2048 if image_date <= camera_switch_date else 1600
    image_h = 2048 if image_date <= camera_switch_date else 1200

    # The length of the edge of the box in pixels.
    # Used 35x35 for a 2048x2048 image
    # Used nnxnn for a 1600x1200 image
    box_edge_len = 35 if image_date <= camera_switch_date else 18

    point_x = lbl_box_obj["point"]["x"]
    point_y = lbl_box_obj["point"]["y"]

    # Calculate xmin and xmax
    xmin = max(0, point_x - 0.5 * box_edge_len)
    xmax = min(image_w, point_x + 0.5 * box_edge_len)

    # Calculate ymin and ymax
    ymin = max(0, point_y - 0.5 * box_edge_len)
    ymax = min(image_h, point_y + 0.5 * box_edge_len)

    return Label(xmin, xmax, ymin, ymax, lbl_box_obj["value"])

# Making a "TFRecordInfo" class to store multiple label classes and some 
# metadata. Since I'm not downloading the images I don't need a lot of these 
# things so I might end up getting rid of this one and pulling the metadata 
# from the local images.
class TFRecordInfo:

    def __init__(self, height, width, filename, source_id, encoded, format, sha_key, labelbox_rowid, labelbox_url, labels):
        self.height = height
        self.width = width
        self.filename = filename
        self.source_id = source_id
        self.encoded = encoded
        self.format = format
        self.sha_key = sha_key
        self.labelbox_rowid = labelbox_rowid
        self.labelbox_url = labelbox_url
        self.labels = labels

    def __repr__(self):
        return "TFRecordInfo({0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8})".format(self.height, self.width, self.filename, self.source_id, type(self.encoded), self.format, self.sha_key, self.labelbox_rowid, self.labels)

# Parsing the labels 
def parse_labelbox_data(data, label_arg):
    records = list()
    image_format = b'jpg'

    for i in range(len(data)):
        record = data[i]

        image_name = record['External ID']

        split_string = image_name.split("_")
        image_date = datetime.strptime(split_string[0], "%Y-%m-%d")

        # Reading in the image from the disk
        # Getting the correct filename. In the future make the image base path an arg
        #filename = record_obj.filename.encode('utf8')
        print(f'Importing {image_name}')
        image_path = path_from_filename(image_name)

        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_jpg = fid.read()

            # Skipping this because the disk image size might eventually be different 
            # from the Labelbox upload size and I want the ratios to be right, so 
            # manually inputting the labelbox upload size below. Can be changed in future.
            # im = Image.open(image_path)
            # width, height = im.size
            #print(f'Image width is: {width}')
            #print(f'Image height is: {height}')


        # Object labels.
        labels = list()
        label_objs = record["Label"]["objects"]

        # Adding them to a nice list using the "label" class defined above.
        # For now I'm only including bounding boxes, so just pollen no tube tips.
        if label_arg == "all":
           for l in label_objs:
               if l.get("bbox"):
                   labels.append(label_from_labelbox_obj(l))
               else:
                   # Makes bounding box from point. Change the bounding bix size
                   # inside the function.
                   labels.append(tube_tip_label_from_labelbox_obj(l, image_date))
        elif label_arg == "pollen":
            for l in label_objs:
                if l.get("bbox"):
                    labels.append(label_from_labelbox_obj(l))
                else:
                    pass
        else: # label_arg == "tube_tip"
            for l in label_objs:
                if l.get("bbox"):
                    pass
                else:
                    # Makes bounding box from point. Change the bounding bix size
                    # inside the fuction.
                    labels.append(tube_tip_label_from_labelbox_obj(l, image_date))

        # Checks the image name to see what camera they came from. If they're from the first camera they get
        # 2048x2048, if they're from the second camera they get 1600x1200

        if image_date <= datetime(2022, 5, 27):
            records.append(
                TFRecordInfo(2048, 2048, image_name, "empty", encoded_jpg,
                             image_format, "empty", "empty", "empty", labels))
        else:
            records.append(
                TFRecordInfo(1200, 1600, image_name, "empty", encoded_jpg,
                             image_format, "empty", "empty", "empty", labels))

    print(f"{len(data)} labeled images parsed")
    
    return data, records

# Getting classes.
def get_classes_from_labelbox(data, label_arg):
    labels_set = set()
    for record in data:
        if isinstance(record, dict):
            if "objects" in record["Label"]:
                label_objs = record["Label"]["objects"]
                for obj in label_objs:
                    if label_arg == "all":
                        labels_set.add(obj["value"])
                    elif label_arg == "pollen":
                        if obj.get("bbox"):
                            labels_set.add(obj["value"])
                    else: # Tube tip classes
                        if obj.get("bbox"):
                            pass # Ignore pollen classes
                        else:
                            labels_set.add(obj["value"])
    labels_list = list(labels_set)
    # Sort labels list so it's the same every time (at least until I add more classes)
    labels_list.sort()
    labels = {}
    for i in range(0, len(labels_list)):
        labels[labels_list[i]] = i
    return labels

def validate_splits(args_splits, parser):
    splits = []

    if not args_splits:
        splits = [100]
    else:
        if any(s <= 0 for s in args_splits):
            parser.error(message='splits must be positive integers')

        if sum(args_splits) < 100:
            splits = args_splits + [100 - sum(args_splits)]
        elif sum(args_splits) > 100:
            parser.error("splits must sum up to <= 100")
        else:
            splits = args_splits

    splits.sort()

    return splits

# Convert a list of percentages adding to 100, ie [20, 30, 50] to a list of indices into the list of records
# at which to split off a new file
def splits_to_record_indices(splits, num_records):
    if splits is None or splits == []:
        splits = [100]
    if sum(splits) != 100: raise ValueError("Percentages must add to 100")
    if not splits or splits == [100]: return [num_records]

    prev_idx = 0
    img_indices = []
    for split_idx,split in enumerate(splits[:-1]):
        end_img_idx = round(prev_idx + (split / 100) * num_records)

        if end_img_idx == prev_idx:
            #Leave in dupes for now. Take out at the end
            pass
        else:
            img_indices += [min(end_img_idx, num_records)]
        prev_idx = end_img_idx

    # Adding the last index this way ensures that it isn't rounded down
    img_indices += [num_records]
    # Dedupe
    return list(OrderedDict.fromkeys(img_indices))

# Get path from filename (for loading images locally according to my directory structure)
def path_from_filename(filename):
    split_string = filename.split("_")

    # Checking to see if it's from the first or second camera
    image_date = datetime.strptime(split_string[0], "%Y-%m-%d")

    if image_date <= datetime(2022, 5, 27):
        print("First camera")
        dir_string = split_string[0] + "_" + split_string[1] + "_" + split_string[2] + "_stab"
        path_base = Path('/media/volume/sdb/jpgs')
        out_path = path_base / dir_string / f'well_{split_string[3]}' / filename
    else:
        print("Second camera")
        dir_string = split_string[0] + "_" + split_string[1] + "_" + split_string[2] + "_normalized_stabilized"
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
        # print(label_obj.xmin)
        # print(record_obj.width)
        # print(label_obj.xmax)
        # print(record_obj.width)
        # print(label_obj.ymin)
        # print(record_obj.height)
        # print(label_obj.ymax)
        # print(record_obj.height)
        xmins.append(label_obj.xmin / record_obj.width)
        xmaxs.append(label_obj.xmax / record_obj.width)
        ymins.append(label_obj.ymin / record_obj.height)
        ymaxs.append(label_obj.ymax / record_obj.height)
        # print("Label_obj.label (classes_text): ", label_obj.label)
        classes_text.append(label_obj.label.encode('utf8'))
        # print(f"Class_dict at label_obj.label with one added is: {class_dict[label_obj.label] + 1}")
        # To match the classes in class_dict
        classes.append(class_dict[label_obj.label] + 1)
        # print(class_dict)
        # print(" ")

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(record_obj.height),
        'image/width': dataset_util.int64_feature(record_obj.width),
        'image/filename': dataset_util.bytes_feature(record_obj.filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(record_obj.source_id.encode('utf8')),
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
    print(len(rearranged_records))
    records = rearranged_records
    print(len(records))
    
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
    data, records = parse_labelbox_data(labels, args.label_type)

    # Getting the different classes present
    class_dict = get_classes_from_labelbox(data, args.label_type)

    splits = validate_splits(args.splits, parser)

    # Making the tfrecords
    generate_tfrecords("add_image_path", args.tfrecord_dest, splits, data, records, class_dict)

if __name__ == "__main__":
    main()
