# Inferring Structural Causal Models with Grammatical Evolution

## Overview

This repository contains the implementation of the methods presented in the paper **"Inferring Structural Causal Models from Data through Grammatical Evolution"**.

The project leverages **grammar-based evolutionary search**, built on top of **PonyGE2**, to synthesize **Structural Causal Models (SCMs)** as **probabilistic programs** directly from data. The framework supports both observational and interventional data, enabling the discovery of causal structures and mechanisms through evolutionary synthesis.

## Requirements

Install the required dependencies with:

```sh
pip install -r requirements.txt
````

All required packages are listed in `requirements.txt`.

## Configuration

The main synthesis parameters are defined in:

```text
algorithm/parameters.py
```

The most relevant parameters are:

* `POPULATION_SIZE` – size of the evolutionary population
* `GENERATIONS` – number of generations
* `EXPERIMENT_NAME` – name of the folder where results will be stored
* `PROGRAM_NAME` – SCM structure to synthesize. Supported benchmark structures are:`chain`,`common_cause`, `common_effect`, `complex`

To define a custom SCM structure, you need to specify in `src/fitness/data_generating_process.py`:

* the set of variables
* the interventions to consider
* the data-generating process used to create the synthetic dataset
* the true and baseline probabilistic programs in SOGA language representing the SCM, for comparison purposes

Additional important parameters:

* `RUNS` – number of runs to execute when using the experiment manager
* `GRAMMAR_FILE` – grammar file located in the `grammar/` folder. Use:`causal_SCM_<n>vars.pybnf` for causal SCM synthesis, `independent_<n>vars.pybnf` for the baseline setting in which all variables are independent
* `INTERVENTIONAL_FITNESS` – set to `True` to include interventional data in the fitness computation, `False` otherwise

Other evolutionary parameters can also be adapted to specific problems, although the benchmark settings used in the paper were kept fixed.

## Datasets

If you want to work with a new SCM structure, you must provide:

* one **observational dataset**
  `scm_name.csv`
* one **interventional dataset** for each intervention
  `scm_name_intervention_variable_value.csv`

These files should be placed in:

```text
src/fitness/datasets
```

To generate synthetic datasets automatically, run:

```text
src/interventions.ipynb
```

Make sure that the intervention list used in the notebook matches the one defined in:

```text
src/fitness/data_generating_process.py
```

## Running the Evolutionary Synthesis

To execute a single synthesis run:

```sh
cd src
python ponyge.py
```

To execute multiple runs in parallel using PonyGE2’s experiment manager:

```sh
cd src
python experiment_manager.py
```

Results are stored in:

```text
results/EXPERIMENT_NAME
```

A separate folder is created for each run. When using the experiment manager, additional `.csv` and `.pdf` files containing run statistics are also generated.

## Plots and Analysis

To reproduce the plots presented in the paper, run:

```text
src/causal_plots_one_obj.ipynb
```

In addition, if you are synthesizing a **3-variable SCM** and set `SAVE_STRUCTURES = True`, you can visualize the percentage of different indirected structures explored during evolution.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{doz2026,
  title={Inferring Structural Causal Models from Data through Grammatical Evolution},
  author={Doz, Romina and Randone, Francesca and Medvet, Eric and Bortolussi, Luca},
  booktitle={...},
  year={2026}
}
```

## Reference

This project builds on **PonyGE2**, the Python implementation of Grammatical Evolution:

Michael Fenton, James McDermott, David Fagan, Stefan Forstenlechner, Erik Hemberg, and Michael O’Neill. 2017. *PonyGE2: Grammatical Evolution in Python*. In *Proceedings of the Genetic and Evolutionary Computation Conference Companion*, 1194–1201.

and on **SOGA** implementation from:

Inference of probabilistic programs with moment-matching gaussian mixtures
F Randone, L Bortolussi, E Incerto, M Tribastone
Proceedings of the ACM on Programming Languages 8 (POPL), 1882-1912, 17	2024

