# MIST: Multimodal Integrated Spatiotemporal Trajectory Estimation

A MATLAB implementation of a non-parametric Expectation-Maximization (EM) algorithm for modeling Alzheimer's disease (AD) progression trajectories using longitudinal biomarker data.

## Overview

This framework jointly estimates subject-specific disease onset ages and population-level biomarker progression trajectories along a latent disease timeline. The method uses monotonic cubic B-spline functions with smoothness constraints to model how biomarkers evolve over the course of Alzheimer's disease.


## Core Methodology

### Model Framework

1. **Biomarker Trajectories**: Monotonic cubic B-spline functions of disease age with integrated squared second-derivative penalty for smoothness
2. **Demographic Effects**: Linear covariates regressed out from biomarker measurements
3. **Disease Onset Ages**: Subject-specific latent variables estimated iteratively
4. **Diagnosis Consistency**: Loss function that enforces agreement between estimated disease age and clinical diagnosis

### EM Algorithm

**E-step**: Update each subject's AD onset age by minimizing the objective function given current trajectory estimates

**M-step**: Estimate trajectory parameters, demographic coefficients, and CN-to-MCI transition point given updated onset ages

The procedure iterates until convergence.

## Main Functions

### `computeNonParaEMFixed`

Core EM algorithm for estimating disease progression trajectories.


### `computeROICurves`

ROI-level biomarker trajectory estimation with fixed disease ages.

### `UsingMultiFeaturesPredictADage`

Predict disease age for new subjects using a pretrained model.

## Usage Examples

### Example 1: Training the Model and Visualizing Trajectories

see demo1

### Example 2: ROI-Level Analysis and Visualization

see demo2

### Example 3: Predicting Disease Age from Cross-Sectional Data

see demo3

## Limitations

- Designed for monotonic progression patterns (Alzheimer's disease)
- Requires longitudinal data with clinical diagnosis information for training
- May not be suitable for other diseases without modification
- Assumes diagnosis codes follow the convention: 1=CN, 2=MCI, 3=Dementia


## Author

**Tianhao Zhang (Zhang Tianhao)**

## Contact

For questions or issues, please contact:
- thzhang@ihep.ac.cn
