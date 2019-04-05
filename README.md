# Calculon

This tool performs semantic segmentation of chest ct scans

Current state of the project (master branch):
[![pipeline status](https://gitlab.com/ebenbuild/calculon/badges/master/pipeline.svg)](https://gitlab.com/ebenbuild/calculon/commits/master) [![coverage report](https://gitlab.com/ebenbuild/calculon/badges/master/coverage.svg)](https://gitlab.com/ebenbuild/calculon/commits/master)


## Installation directions
The use of a virtual environment like [Anaconda](https://www.continuum.io/downloads)
is highly recommended.

After installing Anaconda, create a new environment by running:      
`conda create -n calculon python=3.6`   

This will create a new Python 3.6 environment called  `calculon`.
To activate it run:   
`source activate calculon`

After setting up Anaconda and a new, dedicated development
environment is created and activated as described above, all required third
party libraries can be installed by running:  
`pip install -r requirements.txt`  

To install the calculon for development purposes, use:     
`python setup.py develop`

To uninstall calculon run:  
`python setup.py develop --uninstall`

To update Python packages in your Anaconda environment type:  
`conda update --all`

Get jupyter up and running with Anaconda environment
`python -m ipykernel install --user --name calculon --display-name "Python (calculon)"``

The testing is done through pytest. The testing strategy is more closely described
in [TESTING.md](TESTING.md). To run all tests execute:
`python setup.py test`


## Building the documentation
To be able to build the documentation, an additional package *pandoc* is needed.
Installation instructions are provided [here](http://pandoc.org/installing.html)

Calculon uses sphinx to automatically build a html documentation from docstring.
To build it, navigate into the doc folder and type:    
`cd doc && make html`  

After adding new modules or classes to calculon,
one needs to rebuild the autodoc index by running:    
`sphinx-apidoc -o doc/source calculon -f -M`  
before the make command.
