### PyISV repo

The repository contain the libraries to train a autoencoder-like neural network, used and described in the work https://doi.org/10.48550/arXiv.2407.17924 and https://doi.org/10.1021/acsnano.3c05653

# Folders
- PyISV contains the file with the classes and the functions used:
  - features_calc_utils.py: contains the function to compute the RDF from atomic configurations, being RDF defined as the simple histogram of interatomic distances (histogram is computed via KDE). More in general it contains functions for KDE estimations.
  - network_arxiv.py: contains the 1D convolutional network architecture used in https://doi.org/10.48550/arXiv.2407.17924. The architecture is tuned for an input having a single channel of sinze 340 and expect as flat_dim value 21. In case the input size is changed flat_dim and also the paddings of the decoder layers need to be adjusted to get an output of the same size of the input.
  - network.py: contains an updated 1D convolutional network architecture that seems to offer slightly better performance. This architecture is flexible and can work with inputs with more than one channels. It expect every input channels to be composed by 200 numbers and flat_dim equal to 1. If the input size is changed the decoder padding will need to be adjusted. 
  - train_utils.py: contains additonal classes and functions used in the training of the network

- Scripts: contains example script to run the libraries
  - compute_single_rdf.py: describe how to compute the RDF (histogram of intertomic distances) using the functions contained in PyISV/features_calc_utils.py. It uses ase package and need xyz format of the trajectories. It will compute the overall rdfs of the structures.
  - compute_triple_rdf.py: describe how to compute the RDF (histogram of intertomic distances) using the functions contained in PyISV/features_calc_utils.py. It uses ase package and need xyz format of the trajectories. It will compute 3 RDFs. It is thought for binary systems. The elements will need to be specified in order to have ase library to extract the corresponding positions.
  - model_training_script.py: contains the code to train the network, contains a routine to generate random data to test the code.
  - model_evaluation_script.py: script to run the trained model and evaluate the bottleneck and reconstructions

# Installation
Clone the repo and install it in your environment using:
" pip install . " or " pip install -e . " for editable installation

# Remarks

The library allows for the training of autoencoder like networks. It offers the chance to compute the input descriptors, in particular RDFs starting from xyz files. Anyhow the proposed neural network architectures are general and input descriptors can be changed or computed with other libraries (like scikit-learn for kde) and the networks will work as they are until the input shapes expected are respected. To ensure the correct input shapes some strategies can be used, like zero padding in order to reach the expected size or reinterpolating the 1D input with tools like offered by scientific libraries like scipy.interpolate.interp1d.
If this is not feasible, the decoder architecture and its paddings values will need to be adjusted together with flat_dim in order to ensure a proper output size and a correct forwarding of the network.
Flat_dim is expected to be equal to the size of the final channels of the encoder, since flat_dim times the number of channels of the last encoder layers should match the linear layer size. 
