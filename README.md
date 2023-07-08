# pollen_cv
Quantifying the life of pollen.

## Usage
### process_inference.py
Processes pollen inference data, including tracking with [BayesianTracker](https://github.com/quantumjot/BayesianTracker). Output tracks are optionally visualized with [Napari](https://napari.org/stable/).

To run from Conda, first create the environment then install btrack with pip:
```bash
conda create --name process-inference python=3.8 "cvxopt>=1.2.0" "h5py>=2.10.0" "numpy>=1.17.3" "pooch>=1.0.0" "pydantic>=1.9.0" "scikit-image>=0.16.2" "scipy>=1.3.1" napari pandas scikit-learn -c conda-forge 
```

```bash
conda activate process-inference
pip install btrack
```
