# pollen_cv
Quantifying the life of pollen.

## Usage
### process_inference.py
Processes pollen inference data, including tracking with [BayesianTracker](https://github.com/quantumjot/BayesianTracker). Output tracks are optionally visualized with [Napari](https://napari.org/stable/).

To run from Conda on the UA HPC, first create the environment then install BayesianTracker with pip. At the moment there is a bug with GLIBC which requires manual compilation, but check with the BayesianTrack repo for the latest installation advice.
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
