# bscthesis #

This repository contains my thesis, scripts and data files used there in. 

## Requirements ##
* The implementation is in Python 3.
* [SWIG](http://www.swig.org/download.html) should be installed, to complie the custom library for [Squint](http://jmlr.csail.mit.edu/proceedings/papers/v40/Koolen15a.pdf) 
* LaTeX to complie the thesis (.pdf is also provided)

## Set up ##
1. Get the source code by cloning this git repository
2. Run `make adaboost`, `make nhboost`, `make squintboost` or `make a9a`
  1. (optional) run `python3 generate.py <N> <location>` to generate new data files with the desired size
3. Run `python3 <algorithm>.py <T>` to run an individual test

All scripts are writen with [argparse](https://docs.python.org/3/library/argparse.html) so running `python3 <script> -h` will print the relevant usage information
The scripts can use any datafile writen in [svm light](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html#sklearn.datasets.load_svmlight_file) format

#### How to run tests ####
If you wish to generate plots like in the thesis use `render.py`. The location flags can be left to their default. Run `python3 render.py -h` for more information.
Note that this is very CPU intensive and can take a very long time with large data files. 
