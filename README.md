# Interpretable Lazy Classification with Interval Pattern Structures and Local Interval Explanations

This repository contains the code developed for the paper titled "Interpretable Lazy Classification with Interval Pattern Structures and Local Interval Explanations".

## Abstract
--

## Usage Instructions
The instructions for FCALC and its randomized version can be seen in  [FCALC/README.md](FCALC/README.md). 

The instruction for all other classifiers are here.

### Prerequisites
Ensure you have the necessary Python packages installed. You can install them using pip with the provided `requirements.txt`:

`pip install -r requirements.txt`


### Running Experiments
To reproduce the experimental results discussed in the paper, use `run_experiments.py`. This script allows you to run different experiments:

1. f1: Compute the F1 score of each tested model on each dataset.
2. param_search: Perform hyperparameter search for all tested models.
3. size: Evaluate the size of classifiers for testing local interpretability.
4. time: Measure the training-prediction time and explanation component times.



All results will be stored in the `output` folder. Example outputs can be found in the `example_output` folder.



