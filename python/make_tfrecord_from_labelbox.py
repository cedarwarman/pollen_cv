#!/usr/bin/env python3
import yaml
import io
from pathlib import Path
import labelbox

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

# Parsing the labels (based off https://github.com/caseydaly/LabelboxToTFRecord/)
def parse_labelbox_data(data):
    records = list()
    for i in range(len(data)):
        record = data[i]

        # The original author got the image dimensions from the url, but I 
        # think I will get them when I import the actual image later (the 
        # labelbox images are compressed but probably will be the same size 
        # unless I downscale for the trianing/inference).
        #image_url = record["Labeled Data"]
        #print(image_url)
        image_name = record['External ID']

        # Object labels.
        labels = list()
        label_objs = record["Label"]["objects"]

        # Adding them to a nice list using the "label" class defined above.
        # For now I'm only including bounding boxes, so just pollen no tube tips.
        for l in label_objs:
            if(l.get("bbox")):
                # print(l.get("bbox"))
                labels.append(label_from_labelbox_obj(l))
            else:
                # print("Not a bounding box")
                pass

        # All images uploaded to Labelbox are 2048 by 2048 until further 
        # notice. If necessary I can add the code to download them and get 
        # the actual dimensions be seems like a lot of unnecessary IO for now.




def main():
    # Getting the secrets
    yaml_path = Path.home() / '.credentials' / 'labelbox.yaml'
    labelbox_secrets = open_yaml(yaml_path)
    api_key = labelbox_secrets["api_key"]
    project_id = labelbox_secrets["project_id"]

    # Downloading the labels
    labels = download_labels(api_key, project_id)
    # print("Length of labels is: ", len(labels))
    print(labels[1]["Labeled Data"])

    # Parsing the labels
    parse_labelbox_data(labels)

if __name__ == "__main__":
    main()
