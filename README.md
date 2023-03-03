# pollen_cv
Quantifying the life of pollen.

## Usage
### run_btrack.py
Runs [BayesianTracker](https://github.com/quantumjot/BayesianTracker) on 
pollen tube inference data. Output tracks are visualized with 
[Napari](https://napari.org/stable/).


To run from Conda, first create the environment then install btrack with pip:
```bash
conda create --name btrack python=3.8 "cvxopt>=1.2.0" "h5py>=2.10.0" "numpy>=1.17.3" "pooch>=1.0.0" "pydantic>=1.9.0" "scikit-image>=0.16.2" "scipy>=1.3.1" napari pandas -c conda-forge 
```

```bash
conda activate btrack
pip install btrack
```