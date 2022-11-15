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
#import matplotlib.pyplot as plt
#import scipy.misc
import numpy as np
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
    detections, predictions_dict, shapes = loaded_model(input_tensor)

    # FINISH RETURNS
    return detections

#def make_detections_image(args):
#    """Make and save an image with bounding boxes.
#
#    """
#
#def save_detections(args):
#    """Export a table of detections
#
#    """

def main():
    print("Building model")
    loaded_model = build_model(args.config, args.checkpoint)
    print("Model built")
    
    print("Loading label map")
    label_map_dict, category_index = load_label_map(args.map)
    print("Label map loaded")

    print("Entering inference loop")
    for image_path in pathlib.Path(args.images).glob('*.jpg'):
        print("Running inference on", image_path.name)
        detections = run_inference(loaded_model, image_path)
        print("Detections:", detections)




if __name__ == "__main__":
    main()


















