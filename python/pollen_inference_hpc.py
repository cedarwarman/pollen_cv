#!/usr/bin/env python3.8
"""Pollen inference.

This script performs inference on pollen germination images using a deep learning
model trained with the Tensorflow Object Detection API. This script has been partially
adapted from:
# https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb

Usage:
    python pollen_inference.py \
        --checkpoint [path] \
        --config [path] \
        --map [path] \
        --images [path] \
        --output [path] \
        --save_images \
        --camera both

Arguments:
    --checkpoint
        Path to the model checkpoint.
    --config
        Path to the model config file.
    --map
        Path to the label map file.
    --images
        Path to the directory containing images to do inference on.
    --output
        Path to the directory where the output images will be saved.
    --save_images
        Argument for whether to save images with bounding box annotations.
        If present, images are saved.
    --camera
        Argument for which camera images to run the inference on. Options
        include "both", "one", or "two".

"""


import argparse
import os
import pathlib
from typing import Callable, Dict, Tuple

from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


def parse_arguments(
) -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns
    -------
    parser : argparse.Namespace
        Parsed arguments as a namespace object.

    Raises
    ------
    NotADirectoryError
        If a given directory path is not a directory.
    """

    def dir_path(string: str) -> str:
        if os.path.isdir(string):
            return string
        else:
            try:
                os.makedirs(string, exist_ok=True)
                return string
            except OSError as e:
                raise OSError(f"Error creating directory '{string}': {e}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str, help="Path to model checkpoint file", required=True
    )
    parser.add_argument(
        "--config",
        type=str, help="Path to model config file", required=True
    )
    parser.add_argument(
        "--map",
        type=str, help="Path to the label map file", required=True
    )
    parser.add_argument(
        "--images",
        type=dir_path,
        help="Path to directory containing images to do inference on",
        required=True
    )
    parser.add_argument(
        "--output",
        type=dir_path,
        help="Path to directory where output images will be saved",
        required=True
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save images to output directory if this flag is present"
    )
    parser.add_argument(
        "--camera",
        choices=["both", "one", "two"],
        help="Choose camera whose images you will analyze: both, one, or two",
        required=True
    )

    return parser.parse_args()


def load_image_into_numpy_array(
    path: pathlib.Path
) -> np.ndarray:
    """Load an image from file into a numpy array.
  
    Puts image into numpy array to feed into tensorflow graph. Note that by convention
    we put it into a numpy array with shape (height, width, channels), where
    channels=3 for RGB. Since we're using grayscale images, we copy the grayscale
    values to all channels so the input is as expected.

    Parameters
    ----------
    path : pathlib.Path
        The file path to the image
  
    Returns
    -------
    image_np : np.ndarray
        uint8 numpy array with shape (img_height, img_width, 3)
    """

    with Image.open(path) as image:
        (im_width, im_height) = image.size
        image_np = np.array(image.convert('L')).reshape((im_height, im_width, 1)).astype(np.uint8)
        image_np = np.repeat(image_np, 3, axis=2)

    return image_np


def build_model(
    config_path: str,
    checkpoint_path: str
) -> Callable:
    """Builds a model for inference.

    Loads a model checkpoint and config file and builds the model function for
    inference.

    Parameters
    ----------
    config_path : str
        The path to the config file.
    checkpoint_path : str
        The path to the model checkpoint.

    Returns
    -------
    Callable
        Model detection function based on the config and checkpoint.
    """

    configs = config_util.get_configs_from_pipeline_file(config_path)
    model_config = configs['model']

    detection_model = model_builder.build(
        model_config=model_config, is_training=False
    )
    
    # Load the checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(checkpoint_path).expect_partial()

    @tf.function
    def detect_fn(
        image: tf.Tensor
    ) -> dict:
        """
        Detect objects in image.

        Parameters
        ----------
        image : tf.Tensor
            Input image tensor.

        Returns
        -------
        detections : dict
            A dictionary of the detections.
        """
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections

    return detect_fn


def load_label_map(
    label_map_path: str
) -> Dict[int, str]:
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

    Parameters
    ----------
    label_map_path : str
        The path to the label map file.

    Returns
    -------
    category_index : Dict[int, str]
        A category index.
    """

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return category_index


def run_inference(
    loaded_model: Callable,
    image_path: pathlib.Path
) -> Tuple[np.ndarray, dict]:
    """Run inference on a single image.
    
    Opens an image, converts to numpy array, and runs inference to generate detections.

    Parameters
    ----------
    loaded_model : Callable
        Inference model, output of build_model function.
    image_path : pathlib.Path
        Path to the image to perform inference on.

    Returns
    -------
    Tuple[np.ndarray, dict]
        A tuple containing the image as a numpy array and the detections dictionary.
    """

    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = loaded_model(input_tensor)

    return image_np, detections


def do_non_max_suppression(
    detections: dict
) -> dict:
    """Do non-max suppression.

    Performs the Tensorflow implementation of non-max suppression on inference output.

    Parameters
    ----------
    detections : dict
        Inference output.

    Returns
    -------
    out_dic : dict
        A subset of detections with overlapping boxes removed by non-max suppression.
    """

    # non_max_suppression wants a rank 2 tensor, so removes all 1s
    detection_boxes = tf.squeeze(detections['detection_boxes'])
    detection_scores = tf.squeeze(detections['detection_scores'])

    nms_indices = tf.image.non_max_suppression(
        detection_boxes,
        detection_scores,
        max_output_size=100000,
        iou_threshold=0.5,
        score_threshold=float('-inf'),
        name=None
    )

    # Indexing the input dictionary with the output of non_max_suppression, which is
    # the list of boxes (and score, class) to keep.
    out_dic = {
        'detection_boxes': tf.gather(detection_boxes, nms_indices),
        'detection_scores': tf.gather(detection_scores, nms_indices),
        'detection_classes': tf.gather(tf.squeeze(detections['detection_classes']), nms_indices)
    }

    return out_dic


def get_colors_from_category_index(category_index):
    """Pull out categories and colors.

    Getting the right indices for colors for the
    viz.utils.visualize_boxes_and_labels_on_image_array() function using the category
    index. They will be used for the right track_ids argument.

    Parameters
    ----------
    category_index: Dict[int, Dict[str, str]]
        A dictionary where keys are integers and values are dictionaries containing a
        single key-value pair with key 'name' and value being the category name.

    Returns
    -------
    category_index_unnested : Dict[int, int]
        A dictionary of which category index gets which color index.
    """

    color_index_dict = {
        "aborted": 65,            # Light blue
        "burst": 22,              # Magenta
        "germinated": 5,          # Bright green
        "tube_tip": 32,           # Orange
        "tube_tip_bulging": 34,   # Pale yellow
        "tube_tip_burst": 3,      # Pale red
        "ungerminated": 11,       # Cyan
        "unknown_germinated": 15  # Forest green
    }

    category_index_unnested = {
        key: color_index_dict[category_index[key]['name']]
        for key in category_index
    }

    return category_index_unnested


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

    # Add track_ids to do custom colors:
    # https://github.com/tensorflow/models/blob/cb9b5a1d91d4a6d63881d232721860b3a3f17c43/research/object_detection/utils/visualization_utils.py#L1151
    # https://stackoverflow.com/a/56937982/12312789

    # Getting a dict with the right colors from the category_index:
    color_dict = get_colors_from_category_index(category_index)

    # Makeing the track_id with the color_dict
    track_ids = []
    for detection in (detections["detection_classes"].numpy() + label_id_offset):
        # print(detection)
        # print(color_dict[detection])
        track_ids.append(color_dict[detection])

    # Getting boolean array to remove tube_tip_burst (6, automate this at at some point)
    boolean_array = (detections['detection_classes'].numpy() + label_id_offset).astype(int)
    boolean_array = boolean_array != 6

    # Edits image in place
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'].numpy()[boolean_array],
        (detections['detection_classes'].numpy() + label_id_offset).astype(int)[boolean_array],
        detections['detection_scores'].numpy()[boolean_array],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=800,
        min_score_thresh=.3,
        agnostic_mode=False,
        track_ids=np.array(track_ids)[boolean_array],
        skip_track_ids=True,
        line_thickness=4)

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
    args = parse_arguments()
    print("Building model")
    loaded_model = build_model(args.config, args.checkpoint)
    print("Model built")
    
    print("Loading label map")
    category_index = load_label_map(args.map)
    print("Label map loaded")

    print("Entering inference loop")
    final_df = pd.DataFrame()

    # This version was used for running inference on the full dataset.
    # save_path_dir_string = str(pathlib.Path(args.images).parents[0].name)[:-4] + str(pathlib.Path(args.images).name)[5:]

    # This version was used for running inference on normalized stabilized
    # images in well directories:

    # First, making a directory for the well. It will also make a directory
    # for the experiment, if necessary
    exp_dir = pathlib.Path(args.images).parent.name.replace('_normalized_stabilized', '')
    well_dir = pathlib.Path(args.images).name
    output_dir = pathlib.Path(args.output)

    os.mkdirs(output_dir / exp_dir / well_dir, exist_ok=True)


    for image_path in sorted(pathlib.Path(args.images).glob('*.tif')):
        print("Running inference on", image_path.name)
        image_np, detections = run_inference(loaded_model, image_path)

        print("Doing non-max suppression")
        detections = do_non_max_suppression(detections)

        print("Making image")
        out_image = make_detections_image(image_np, detections, category_index)

        print("Saving image")
        image_path_string = str(image_path.stem)[:-5] + '_inference.tif'
        out_image.save(output_dir / exp_dir / well_dir / image_path_string)

        print("Extracting detections")
        detections_table = get_detections(detections, category_index, image_path.stem)
        final_df = pd.concat([final_df, detections_table]).reset_index(drop = True)

    print("Saving detections")
    # This version for running it on the full set
    # final_df.to_csv(save_path / (str(image_path.stem) + '_predictions.tsv'),
    #     index = False,
    #     sep = '\t')

    # This version for running it on the training and validation sets 
    final_df.to_csv(
        output_dir / 
        exp_dir / 
        "predictions"
        / (str(exp_dir) + "_" + str(well_dir) + "_predictions.tsv"),
        index=False,
        sep="\t",
    )

    print("Finished")

if __name__ == "__main__":
    main()

