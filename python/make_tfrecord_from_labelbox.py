#!/usr/bin/env python3

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

from object_detection.utils import dataset_util

# Opening the Labelbox secrets
def open_yaml(yaml_path):
    with open(yaml_path) as file:
        yaml_dict = yaml.safe_load(file)
    return yaml_dict

# Downloading the labels
def download_labels(api_key, project_id):
    # Enter your Labelbox API key here
    LB_API_KEY = api_key

    # Create Labelbox client
    lb = labelbox.Client(api_key=LB_API_KEY)

    # Get project by ID
    project = lb.get_project(project_id)

    # Export image and text data as an annotation generator:
    labels = project.label_generator()
    
    # Export labels created in the selected date range as a json file:
    labels = project.export_labels(download = True) 

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
def parse_labelbox_data(data):
    records = list()
    image_format = b'jpg'

    for i in range(len(data)):
        record = data[i]

        image_name = record['External ID']

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
        for l in label_objs:
            if l.get("bbox"):
                # print(l.get("bbox"))
                labels.append(label_from_labelbox_obj(l))
            else:
                # print("Not a bounding box")
                pass

        # All images uploaded to Labelbox are 2048 by 2048 until further 
        # notice. If necessary I can add the code to download them and get 
        # the actual dimensions be seems like a lot of unnecessary IO for now.
        # I will add the other things as I need them.
        records.append(TFRecordInfo(2048, 2048, image_name, "empty", encoded_jpg, image_format, "empty", "empty", "empty", labels))

    print(f"{len(data)} labeled images parsed")
    
    return data, records

# Getting classes.
def get_classes_from_labelbox(data):
    labels_set = set()
    for record in data:
        if isinstance(record, dict):
            if "objects" in record["Label"]:
                label_objs = record["Label"]["objects"]
                for obj in label_objs:
                    # This version does only bounding boxes
                    if obj.get("bbox"):
                        labels_set.add(obj["value"])
    labels_list = list(labels_set)
    labels = {}
    for i in range(0, len(labels_list)):
        labels[labels_list[i]] = i
    return labels

def validate_splits(args_splits):
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
    dir_string = split_string[0] + "_" + split_string[1] + "_" + split_string[2] + "_stab" 
    # Will change, consider arg later
    path_base = Path('/media/volume/sdb/jpgs')
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
        print(label_obj.xmin)
        print(record_obj.width)
        print(label_obj.xmax)
        print(record_obj.width)
        print(label_obj.ymin)
        print(record_obj.height)
        print(label_obj.ymax)
        print(record_obj.height)
        xmins.append(label_obj.xmin / record_obj.width)
        xmaxs.append(label_obj.xmax / record_obj.width)
        ymins.append(label_obj.ymin / record_obj.height)
        ymaxs.append(label_obj.ymax / record_obj.height)
        print("Label_obj.label: ", label_obj.label)
        classes_text.append(label_obj.label.encode('utf8'))
        print(class_dict[label_obj.label])
        classes.append(class_dict[label_obj.label])
        print(" ")

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

# Making the tfrecords with split
def generate_tfrecords(image_source, tfrecord_dest, splits, data, records, class_dict):
    print("Making tfrecords")

    # Hopefully from here it's pretty straightforward? Just need to add the 
    # images as an arg and deal with their paths in parse_labelbox_data and/or
    # create_tf_example, figure out what's actually necessary in create_tf_example,
    # and write it up.

    # Making a directory for the output tfrecords
    tfrecord_folder = tfrecord_dest
    print(tfrecord_folder)
    if not os.path.exists(tfrecord_folder):
        os.makedirs(tfrecord_folder)
    # Might not need to do this if I'm using pathlib
    if tfrecord_folder[len(tfrecord_folder)-1] != '/':
        tfrecord_folder += '/'

    strnow = datetime.now().strftime('%Y-%m-%d_t%H%M')
    splits = splits_to_record_indices(splits, len(records))
    print("splits is: ", splits)
    assert splits[-1] == len(records), f'{splits}, {len(records)}'

    random.shuffle(records)
    
    # Making the tfrecord files
    split_start = 0
    print(f'Creating {len(splits)} TFRecord files:')
    for split_end in splits:
        outfile = f'{strnow}_{split_end - split_start}.tfrecord'
        print(f"Outfile is: {outfile}")
        outpath = tfrecord_folder + outfile
        with tf.io.TFRecordWriter(outpath) as writer:
            for record in records[split_start:split_end]:
                #### STOPPED HERE ####
                # The create_tf_example function is going to need some modification
                tf_example = create_tf_example(record, class_dict)
                writer.write(tf_example.SerializeToString())
        print('Successfully created TFRecord file at location: {}'.format(outpath))
        split_start = split_end

    pb_file_name = f'{strnow}.pbtxt'
    #### CHECK THIS ONE ####
    label_text = class_dict_to_label_map_str(class_dict)
    with open(tfrecord_folder + pb_file_name, 'w') as label_file:
        print(f'Creating label map file')
        label_file.write(label_text)

def main():
    # Some args
    parser = argparse.ArgumentParser(description='Convert Labelbox data to TFRecord and store .tfrecord file(s) locally.')
    parser.add_argument('--tfrecord-dest', help="Destination folder for .tfrecord file(s)", default="tfrecord")
    parser.add_argument('--splits', help="Space-separated list of integer percentages for splitting the " +
        "output into multiple TFRecord files instead of one. Sum of values should be <=100. " +
        "Example: '--splits 10 70' will write 3 files with 10%%, 70%%, and 20%% of the data, respectively",
        nargs='+',
        type=int,
    )
    args = parser.parse_args()

    # Getting the secrets
    yaml_path = Path.home() / '.credentials' / 'labelbox.yaml'
    labelbox_secrets = open_yaml(yaml_path)
    api_key = labelbox_secrets["api_key"]
    project_id = labelbox_secrets["project_id"]

    # Downloading the labels
    labels = download_labels(api_key, project_id)

    # Parsing the labels
    data, records = parse_labelbox_data(labels)

    # Getting the different classes present
    class_dict = get_classes_from_labelbox(data)

    splits = validate_splits(args.splits)

    # Making the tfrecords
    generate_tfrecords("add_image_path", args.tfrecord_dest, args.splits, data, records, class_dict)

if __name__ == "__main__":
    main()
