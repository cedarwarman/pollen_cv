# Pollen CV
Quantifying the life of pollen using deep learning and multiple object tracking.

## Usage

### Model training
Models are trained using the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). For this project, I used the CenterNet Hourglass-104 architecture. Training takes place using the `bash/train_centernet_hourglass104_1024x1024.sh` script, while evaluation is done with the `bash/val_centernet_hourglass104_1024x1024.sh` script. These scripts were run inside a Docker container (`docker/Dockerfile`) on a [Jetstream2](https://jetstream-cloud.org/index.html) VM. Training was done using an Nvidia A100 GPU.

### Inference
Inference on pollen images is done with the `python/pollen_inference_hpc.py` script using the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). It is designed to be run using Slurm at the University of Arizona HPC (`slurm/hpc_inference_array.slurm`). The job is arrayed in batches to maximize GPU use efficiency and runs in a Docker container through Apptainer. The container is described in `docker/Dockerfile`. Inference was done on using 4 Nvidia V100 GPUs.

Here is example inference output for a single image. Predicted pollen grain and tube tip locations are marked with bounding boxes.
![Example inference output](/data/img/pollen_object_detection_output.jpg)

### Processing raw inference
Raw inference outputs are processed using `python/process_inference.py`, including tracking with [BayesianTracker](https://github.com/quantumjot/BayesianTracker). Output tracks are optionally visualized with [Napari](https://napari.org/stable/).

To run from Conda on the University of Arizona HPC, first create the environment then install BayesianTracker with pip. At the moment there is a bug with GLIBC which requires manual compilation, but check with the BayesianTrack repo for the latest installation advice.
```bash
conda create --name process-inference python=3.8 napari pandas scikit-learn
```

```bash
conda activate process-inference
git clone -b fix-makefile https://github.com/quantumjot/btrack.git
cd btrack
./build.sh
pip install -e .
```

Here is an example of multiple object tracking output. Predicted tracks for linked bounding boxes are separated by color.
![Multiple object tracking example](/data/img/multiple_object_tracking.jpg)
