from setuptools import setup, find_packages

setup(
    name='pyisv',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pyyaml',
        'numpy',
        'torch',
        'torchsummary',
        'scikit-learn',
        'ase',
        'tqdm'
    ],
    author='Emanuele Telari, Fabio Rasera',
    author_email='telariemanuele@gmail.com, fabiorasera00@gmail.com',
    description='Repository for implementing the methods presented in https://doi.org/10.48550/arXiv.2407.17924 and  https://doi.org/10.1021/acsnano.3c05653 ',
    url='https://github.com/etela995/PyISV.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
