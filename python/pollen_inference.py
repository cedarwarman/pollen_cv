#!/usr/bin/env python3

# This script has been partially adapted from:
# https://github.com/tensorflow/models/blob/master/research/object_detection/
# colab_tutorials/inference_tf2_colab.ipynb

import io
import os
import glob
import pathlib
import argparse
import matplotlib
import matplotlib.pyplot as plt
#import scipy.misc
import numpy as np
import pandas as pd
from six import BytesIO # Part of io but uses six for Python compatibility
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

### Adding arguments
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help = "Path to model checkpoint", required = True)
parser.add_argument('--config', type=str, help = "Path to model config file", required = True)
parser.add_argument('--map', type=str, help = "Path to the label map file", required = True)
parser.add_argument('--images', type=dir_path, help = "Path to directory containing images to do inference on", required = True)
parser.add_argument('--output', type=dir_path, help = "Path to directory where output images will be save", required = True)
args = parser.parse_args()

### Functions
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
  
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    CHANGED TO channels=1 FOR GRAYSCALE, CHECK TO SEE IF THIS BREAKS THINGS
  
    Args:
      path: the file path to the image
  
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """

    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape(
        (im_height, im_width, 1)).astype(np.uint8)
    # Copying channels to 3 so input matches (maybe fix this in the model 
    # somewhere to make it faster?) Also make sure that this is what it 
    # did during training. 
    # image_np = np.tile(image_np, (1, 1, 3)).astype(np.uint8)
    image_np = np.tile(image_np, (1, 1, 3))

    return image_np

def build_model(config_path, checkpoint_path):
    """Build a model for inference.

    Loads a model checkpoint and config file and builds the model for 
    inference.

    Args:
        config_path: the path to the config file
        checkpoint_path: the path to the model checkpoint

    Returns:
        model detection function based on the config and checkpoint
    """

    # Somehow this gets the right config
    configs = config_util.get_configs_from_pipeline_file(config_path)
    model_config = configs['model']

    # Load pipeline config and build a detection model
    detection_model = model_builder.build(
          model_config = model_config, is_training = False)
    
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(checkpoint_path).expect_partial()
    
    def get_model_detection_function(model):
      """Get a tf.function for detection."""
    
      @tf.function
      def detect_fn(image):
        """Detect objects in image."""
    
        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)
    
        return detections, prediction_dict, tf.reshape(shapes, [-1])
    
      return detect_fn

    detect_fn = get_model_detection_function(detection_model)

    return detect_fn

def load_label_map(label_map_path):
    """Load label map.

    Loads label map for plotting. Should look like this (starts at 1):
    item {
        id: 1
        name: 'Cat'
    }
     
    item {
        id: 2
        name: 'Dog'
    }

    Args:
        label_map_path: the path to the label map file

    Returns:
        a tuple containing a label map dictionary and a category index
    """

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    # It looks like the plotting doesn't even use label_map_dict, so maybe 
    # in the future.
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    return label_map_dict, category_index

def run_inference(loaded_model, image_path):
    """Run inference on a single image.
    
    Opens an image, converts to numpy array, and runs inference to generate 
    detections.

    Args:
        loaded_model: inference model, output of build_model function
        image_path: path to image to perform inference on

    Returns:
        a tuple containing the image as an numpy array, ?detections?, ?predictions_dict?, ?shapes?
    """

    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    # Might not needs predictions_dict or shapes
    detections, predictions_dict, shapes = loaded_model(input_tensor)

    return image_np, detections

def do_non_max_suppression(detections):
    """Do non-max suppression

    Performs the Tensorflow implementation of non-max suppression on inference 
    output.

    Args:
        detections: inference output

    Returns:
        a subset of detections with overlapping boxes removed by non-max 
        suppression
    """
    print(detections['detection_boxes'])
    print(detections['detection_scores'])
    print(detections['detection_classes'])
    # nmsed_boxes, nmsed_scores, nmsed_classes = tf.image.combined_non_max_suppression(
    #     tf.expand_dims(detections['detection_boxes'], 2),
    #     detections['detection_scores'],
    #     10000,
    #     10000,
    #     iou_threshold=0.5,
    #     score_threshold=float('-inf')
    # )

    # print(nmsed_boxes)
    # print(nmsed_scores)
    # print(nmsed_classes)
    # exit()

    nms_vec = tf.image.non_max_suppression(
        tf.squeeze(detections['detection_boxes']), # Wants rank 2 tensor, so removes all 1s
        tf.squeeze(detections['detection_scores']), # Same
        100000,
        iou_threshold=0.5,
        score_threshold=float('-inf'),
        name=None)

    # Indexing the input dictionary with the output of non_max_suppression,
    # which is the list of boxes (and score, class) to keep.
    print(nms_vec)
    out_dic = detections.copy()

    out_dic['detection_boxes'] = tf.gather(tf.squeeze(detections['detection_boxes']), nms_vec)
    out_dic['detection_scores'] = tf.gather(tf.squeeze(detections['detection_scores']), nms_vec)
    out_dic['detection_classes'] = tf.gather(tf.squeeze(detections['detection_classes']), nms_vec)
    
    print("Did nms")
    print(out_dic['detection_boxes'])
    print(out_dic['detection_scores'])
    print(out_dic['detection_classes'])
    
    return(out_dic)

def make_detections_image(image_np, detections, category_index):
    """Make and save an image with bounding boxes.

    Using an image and detections, plots detections on the image.

    Args:
        image_np: numpy array of image, output of run_inference
        detections: bounding box output of run_inference

    Returns:
        PIL-formatted image 
    """
    label_id_offset = 1

    # Edits image in place
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'].numpy(),
        (detections['detection_classes'].numpy() + label_id_offset).astype(int),
        detections['detection_scores'].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=800,
        min_score_thresh=.05,
        agnostic_mode=False)

    # Convert to PIL format for saving
    output_image = Image.fromarray(image_np)

    return output_image

def get_detections(detections, category_index, image_name):
    """Makes a table of detections

    Reformats inference output into a usable structure for downstream object 
    tracking.

    Args:
        detections: dictionary of predictions output from inference
        category_index: links inference numbers to the actual object categories
        image_name: name of the image for metadata

    Returns:
        Pandas data frame of all the detected objects in a single image
    """

    # Getting items out of the detections dictionary
    detection_boxes = detections['detection_boxes'].numpy() # shape=
    detection_scores = detections['detection_scores'].numpy() # shape=
    detection_classes = detections['detection_classes'].numpy() # shape=

    # Making the data frames (probably a more elegant way to do this)
    df = pd.DataFrame({
        'score':detection_scores,
        'class':detection_classes})
    box_df = pd.DataFrame(detection_boxes, columns = ['ymin', 'xmin', 'ymax', 'xmax'])
    output_df = pd.concat([df, box_df], axis = 1)

    # Replacing the class index with the class name. First I need to unnest the 
    # cateogry_index dictionary. 
    class_dict = {}
    for key in category_index:
        class_dict[category_index[key].get('id')] = category_index[key].get('name')

    # Changing class to int and adding the label offset that object detection does
    output_df['class'] = output_df['class'].astype(int)
    output_df['class'] = output_df['class'] + 1

    # Doing the replacement
    output_df = output_df.replace({"class": class_dict})

    # Adding some metadata
    name_list = image_name.split('_')
    output_df['date'] = name_list[0]
    output_df['run'] = name_list[1][3:]
    output_df['tempc'] = name_list[2][:-1]
    output_df['well'] = name_list[3]
    output_df['timepoint'] = name_list[4][1:]

    output_df = output_df[['date', 'run', 'well', 'timepoint', 'tempc', 
        'class', 'score', 'ymin', 'xmin', 'ymax', 'xmax']]

    return output_df

def main():
    print("Building model")
    loaded_model = build_model(args.config, args.checkpoint)
    print("Model built")
    
    print("Loading label map")
    label_map_dict, category_index = load_label_map(args.map)
    print("Label map loaded")

    print("Entering inference loop")
    final_df = pd.DataFrame()
    save_path_dir_string = str(pathlib.Path(args.images).parents[0].name)[:-4] + str(pathlib.Path(args.images).name)[5:]
    save_path = pathlib.Path(args.output) / save_path_dir_string
    os.mkdir(save_path)

    for image_path in sorted(pathlib.Path(args.images).glob('*.jpg')):
        print("Running inference on", image_path.name)
        image_np, detections = run_inference(loaded_model, image_path)

        print("Doing non-max suppression")
        detections = do_non_max_suppression(detections)

        print("Making image")
        out_image = make_detections_image(image_np, detections, category_index)

        print("Saving image")
        out_image.save(save_path / (str(image_path.stem) + '_inference.jpg'))

        print("Extracting detections")
        detections_table = get_detections(detections, category_index, image_path.stem)
        final_df = pd.concat([final_df, detections_table]).reset_index(drop = True)

    print("Saving detections")
    final_df.to_csv(save_path / (str(image_path.stem) + '_predictions.tsv'),
        index = False,
        sep = '\t')

    print("Finished")




if __name__ == "__main__":
    main()


















