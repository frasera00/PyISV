# PyISV: Autoencoder-based Convolutional Neural Network for low-dimensional representation and classification of Atomic structures

PyISV is a Python library designed for training autoencoder-like neural networks to find low dimensional representations of metal nanoclusters. It includes tools for computing Radial Distribution Functions (RDFs) and provides flexible neural network architectures for various input descriptors. This library is used and described in the following works:

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
Stores pre-trained models, such as `classifier_best.pt`, for classification of structures for Ag38 clusters.

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

## Usage Examples

```
For more examples, refer to the `scripts/` directory or the Jupyter notebooks in the `notebooks/` directory.

### 1. Extracting RDFs from Structure Data
To compute Radial Distribution Functions (RDFs) from atomic structure data, use the `generate_RDFs.py` script located in the `scripts/` folder. Below is an example usage:

1. Open the `generate_RDFs.py` script.
2. Modify the following parameters as needed:
   - `XYZ_PATH`: Path to the XYZ file containing atomic structures.
   - `LABEL_FILE`: Path to the label file.
   - `OUTPUT_DIR`: Directory to save the RDFs.
   - `MIN_DIST`, `MAX_DIST`, `N_BINS`, `BANDWIDTH`: Parameters for RDF calculation.
   - `FRACTION`, `PERIODIC`, `REPLICATE`: Additional options for RDF computation.
3. Run the script:

```bash
python scripts/generate_RDFs.py
```

The script will log the computation process in `rdf_computation.log` within the output directory.

### 2. Training an Autoencoder for Dimensionality Reduction
To train an autoencoder, use the `train_autoencoder.py` script. Below is an example workflow:

1. Ensure the `config_autoencoder.yaml` file is properly configured:
   - `input`: Specify paths for input data (`data/RDFs/rdf_images.pt`).
   - `model`: Define the model architecture, including encoder and decoder channels, and embedding dimensions.
   - `training`: Set training parameters like batch size, learning rate, and epochs.
   - `output`: Specify paths for saving logs, models, and normalization parameters.

2. Run the script:

```bash
python scripts/train_autoencoder.py
```

This will train an autoencoder model using the RDFs specified in the configuration file. The best model and training logs will be saved in the paths defined in `config_autoencoder.yaml`.

### 3. Training a Classification Network
To train a classification network using RDFs and their corresponding labels, use the `train_classification.py` script. Below is an example workflow:

1. Ensure the `config_classification.yaml` file is properly configured:
   - `input`: Specify paths for input data (`example file: data/RDFs/rdf_images.pt`) and labels (`example file: data/Ag38_labels/labels.pt`).
   - `model`: Define the model architecture, including the number of classes and encoder channels.
   - `training`: Set training parameters like batch size, learning rate, and epochs.
   - `output`: Specify paths for saving logs, models, and normalization parameters.

2. Run the script:

```bash
python scripts/train_classification.py
```

This will train a classification model using the RDFs and labels specified in the configuration file. The best model and training logs will be saved in the paths defined in `config_classification.yaml`.

## Remarks
The library allows for the training of autoencoder-like networks and the computation of input descriptors (e.g., RDFs). While the proposed neural network architectures are general, input descriptors can be customized. Ensure that input shapes match the expected dimensions by using strategies like zero-padding or interpolation.

For more details, refer to the source code and the referenced publications.
