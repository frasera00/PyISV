### PyISV repo

The repository contain the libraries to train a autoencoder-like neural network, used and described in the work https://doi.org/10.48550/arXiv.2407.17924

# Folders
- PyISV contains the file with the classes and the functions used:
  - kde_rdf.py: contains the function to compute the RDF from atomic configurations, being RDF defined as the simple histogram of interatomic distances (istogram is computed via KDE)
  - network.py: contains the network architecture used in https://doi.org/10.48550/arXiv.2407.17924
  - train_utils.py: contains additonal classes and functions used in the training of the network

- ExampleScripts: contains example script to run the libraries
  - rdf_calc_script.py: describe how to compute the RDF (histogram of intertomic distances) using the functions contained in PyISV/kde_rdf.py
  - model_training_script.py: contains the code to train the network
  - model_evaluation_script.py: script to run the trained model and evaluate the bottleneck and reconstructions

# Installation
Clone the repo and install it in your environment using:
pip install . or pip install -e . 
