# Multi-Objective Parallel Local Surrogate-Assisted Search (MOPLS)

## Introduction
Multi-objective Optimization Problems (MOPs), which aim to find optimal trade-off solutions regarding to multiple conflicting and equally important objectives, commonly exist in real-world applications. Without loss of generality, an MOP is mathematically defined by

$$
\min_{\mathbf{x} \in \mathcal{D}} \mathbf{f}(\mathbf{x}) = \[f_1(\mathbf{x}), \dots, f_k(\mathbf{x})\]^T
$$

where $\mathbf{x}$ denotes a vector composed of $d$ decision variables. The search space $\mathcal{D}$ denotes a hypercube in $\mathbb{R}^d$ and it is bounded by a lower bound $\mathbf{l}$ and a upper bound $\mathbf{u} \in \mathbb{R}^d$. $\mathbf{f}$ is composed of $k$ objective functions with $f_i : \mathbb{R}^d \rightarrow \mathbb{R}$ representing the $i$-th objective to be optimized, $i = 1, \dots, k$.

**MOPLS** is a surrogate-assisted algorithm designed for computationally expensive multi-objective optimization problems where each objective is assumed to be black-box and expensive-to-evaluate. In each iteration, MOPLS incorporates a tabu mechanism to dynamically determine new points for expensive evaluations via a series of independent surrogate-assisted local searches. The master-worker architecture in MOPLS allows the algorithm to conduct either synchronous or asynchronous parallel processing with multiple processors.

## Installation

The Python version of MOPLS is implemented upon a surrogate optimization toolbox, pySOT, which provides various types of surrogate models, experimental desings, acquisition functions, and test problems. To find out more about pySOT, please visit its [toolbox documentation](http://pysot.readthedocs.io/) or refer to the corresponding paper [David Eriksson, David Bindel, Christine A. Shoemaker. pySOT and POAP: An event-driven asynchronous framework for surrogate optimization. arXiv preprint arXiv:1908.00420, 2019](https://doi.org/10.48550/arXiv.1908.00420). 

In a virtual environment with Python 3.4 or newer, the instructions for installation of pre-requisites is as follows:

```
pip install pySOT
pip install matplotlib
```

## Running MOPLS

An example of how to run MOPLS on a bi-objective DTLZ2 is provided in the file experiments.py. The setup for running the algorithm is synonymous to how optimization experiments are setup in pySOT. 

* Note: The MOPLS package have already implemented some problems for testing. You can also easily design your own test problem classes by inheriting from the OptimizationProblem class defined in pySOT. Kindly look at how problems are defined in the multiobjective_problems.py.
* Note: Various types of experimental design methods and surrogate models are available in pySOT. Like the customization for test problem, you can also program and use your own exp_design and surrogate classes in RECAS, but they must inherit from the parent classes, pySOT.ExperimentalDesign and pySOT.Surrogate, respectively.

For further information please contact the developer at [email](mailto:wangwenyu0928@gmail.com).

## Reference

If you use MOPLS, please cite the following paper:
```
@article{wang2023efficient,
  title={Efficient multi-objective optimization through parallel surrogate-assisted local search with tabu mechanism and asynchronous option},
  author={Wang, Wenyu and Akhtar, Taimoor and Shoemaker, Christine A},
  journal={Engineering Optimization},
  pages={1--17},
  year={2023},
  publisher={Taylor \& Francis}
}
```
