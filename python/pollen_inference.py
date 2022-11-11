#!/usr/bin/env python3

# This script has been partially adapted from:
# https://github.com/tensorflow/models/blob/master/research/object_detection/
# colab_tutorials/inference_tf2_colab.ipynb

import io
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
parser.add_argument('--checkpoint', type=dir_path, help = "Path to model checkpoint", required = True)
parser.add_argument('--config', type=dir_path, help = "Path to model config file", required = True)
parser.add_argument('--images', type=dir_path, help = "Path to directory containing images to do inference on", required = True)
parser.add_argument('--map', type=dir_path, help = "Path to the label map file", required = True)

### Functions
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
  
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
  
    Args:
      path: the file path to the image
  
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

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
    # Load pipeline config and build a detection model
    detection_model = model_builder.build(
          model_config = config_path, is_training = False)
    
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
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    return label_map_dict, category_index

def run_inference(args):
    """Run inference on a single image.
    
    """

def make_detections_image(args):
    """Make and save an image with bounding boxes.

    """

def save_detections(args):
    """Export a table of detections

    """




















