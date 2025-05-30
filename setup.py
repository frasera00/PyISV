from setuptools import setup, find_packages

setup(
    name='pyisv',
    version='2.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchsummary',
        'scikit-learn',
        'ase',
        'tqdm',
        'openTSNE',
    ],
    author='Emanuele Telari, Fabio Rasera',
    author_email='telariemanuele@gmail.com, fabiorasera00@gmail.com',
    description='Repository for implementing the methods presented in https://doi.org/10.48550/arXiv.2407.17924 and  https://doi.org/10.1021/acsnano.3c05653 ',
    url='https://github.com/frasera00/PyISV.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
