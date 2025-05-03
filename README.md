# PyISV: Autoencoder-Based Neural Network for Atomic Configurations

PyISV is a Python library designed for training autoencoder-like neural networks to analyze atomic configurations. It includes tools for computing Radial Distribution Functions (RDFs) and provides flexible neural network architectures for various input descriptors. This library is used and described in the following works:

- [DOI: 10.48550/arXiv.2407.17924](https://doi.org/10.48550/arXiv.2407.17924)
- [DOI: 10.1021/acsnano.3c05653](https://doi.org/10.1021/acsnano.3c05653)

## Features
- Compute RDFs from atomic configurations using Kernel Density Estimation (KDE).
- Train autoencoder-like neural networks with flexible architectures.
- Evaluate trained models for bottleneck and reconstruction performance.

## Updated Folder Structure

### PyISV
Contains the core library files for the project.

### scripts
Contains Python scripts for generating RDFs, training models, and evaluating models.

### notebooks
Contains Jupyter notebooks for interactive exploration and predictions.

### data
Stores datasets, predictions, and RDF-related data.

### tests
Contains unit tests for the library.

### models
Stores pre-trained models, such as `classifier_best.pt`.

## Installation
Clone the repository and install it in your Python environment:

```bash
pip install .
```

For editable installation:

```bash
pip install -e .
```

## Getting Started
1. Install the library as described above.
2. Use the example scripts in the `scripts/` folder to compute RDFs or train models.
3. Refer to the `PyISV` folder for detailed implementations of the library's features.

## Updated Usage Example

Here is an example of how to compute RDFs using the reorganized structure:

```python
from PyISV.features_calc_utils import compute_single_rdf

# Example usage
rdf = compute_single_rdf("data/Ag38_labels/isv_coords_2D_nonMin_to_min_nCu_0.txt")
print(rdf)
```

For more examples, refer to the `scripts/` directory or the Jupyter notebooks in the `notebooks/` directory.

## Remarks
The library allows for the training of autoencoder-like networks and the computation of input descriptors (e.g., RDFs). While the proposed neural network architectures are general, input descriptors can be customized. Ensure that input shapes match the expected dimensions by using strategies like zero-padding or interpolation.

For more details, refer to the source code and the referenced publications.
