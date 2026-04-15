## Overview
This repository contains the implementation of the methods described in the paper **"Inferring Structural Causal Models from Data through Grammatical Evolution"**. It exploits grammar-based evolution (by using PonyGE2) to synthesize SCMs in the form of probabilistic programs from data.

## Requirements
To run the code, install the dependencies reported in the file requirements.txt

## Usage

#### Parameters:

The parameters of the synthesis process are set in the file algorithm/parameters.py. You should take care of the following:

- POPULATION_SIZE: Population size
- GENERATIONS: Number of generations of the evolutionary process
- EXPERIMENT_NAME: Name of the folder where you want to store the results
- PROGRAM_NAME: The SCM structure you aim to synthesize. You can use "chain", "common_cause", "common_effect", or "complex" to reproduce the results of the paper. If you want to try your own structure, you need to specify in src/fitness/data_generating_process.py the set of interventions, the set of variables involved, the data generating process (for generating the synthetic dataset), the true and the baseline probabilistic programs in soga language representing your SCM (for the comparisons).
- RUNS: if you are using the experiment manager
- GRAMMAR_FILE: name of the file causal_SCM_**n**vars.pybnf contained in the folder "grammar", where **n** is the number of variables involved in the SCM you aim to synthesize. For the baseline evolution in which all the variables are independent, use independent_**n**vars.pybnf
- 'INTERVENTIONAL_FITNESS': True for including interventions in the computation of the fitness, False otherwise

Other evolutionary parameters can be changed accordingly to the specific problems, but for the benchmark problems they were fixed. 

#### Datasets:
If you are working with a new structure, you need to provide an observational dataset (scm_name.csv) and a dataset for each intervention (scm_name_intervention_variable_value.csv) in the folder src/fitness/datasets. To generate the synthetic data automatically, run the notebook src/interventions.ipynb with the interventions_list you would like to use (they must be the same of rc/fitness/data_generating_process.py). 

### Running the Evolutionary Synthesis
To execute the main synthesis process, run:

```sh
cd src
python ponyge.py
```
To execute multiple runs of the synthesis process, in a multicore framework, use the experiment manager provided by PonyGE2:

```sh
cd src
python experiment_manager.py
```

You can find results in folder results/EXPERIMENT_NAME: one folder is created for each run, if you use the experiment manager, you get also a set of .csv and .pdf files with the statistics of the runs.

### Plots

In order to get all the plots of the paper, run the notebook src/causal_plots_one_obj.ipynb. If you are synthesizing an SCM with 3 variables and setting the parameter SAVE_STRUCTURES to True, you can also see a plot with the percentage of each kind of indirected structures during the evolution.

## Citation
If you use this code, please cite our paper:

```bibtex
@inproceedingsdoz2025evolutionary,
  title={Inferring Structural Causal Models from Data through Grammatical Evolution},
  author={Doz, Romina and Randone, Francesca and Medvet, Eric and Bortolussi, Luca},
  booktitle={...},
  year={2026}
}
```
## Reference

Michael Fenton, James McDermott, David Fagan, Stefan Forstenlechner, Erik
Hemberg, and Michael O’Neill. 2017. Ponyge2: Grammatical evolution in python.
In Proceedings of the Genetic and Evolutionary Computation Conference Companion.
1194–1201
