<<<<<<< HEAD

## Overview
This repository contains the implementation of the methods described in the paper **"Evolutionary Synthesis of Probabilistic Programs"**. It exploits grammar-based evolution (by using PonyGE2) to synthesize probabilistic programs from data, in order to learn the data generating process represented by a probabilistic program. 

## Requirements
To run the code, install the dependencies reported in the file requirements.txt

## Usage

#### Parameters:

The parameters of the synthesis process are set in the file algorithm/parameters.py. You should take care of the following:

- POPULATION_SIZE: Population size
- GENERATIONS: Number of generations of the evolutionary process
- EXPERIMENT_NAME: Name of the folder where you want to store the results
- PROGRAM_NAME: The program you aim to synthesize, which has to be specified in baselines/PROGRAM_NAME.soga for comparison. Moreover, in fitness/data_generating_process.py the data generating process related to the process "PROGRAM_NAME" has to be described in order to generate data during the synthesis process.
- RUNS: if you are using the experiment manager
- GRAMMAR_FILE: name of the file .pybnf contained in the folder "grammar". The program structure (skecth) can be set in the <fc> line, the variable names can be specified in the <idv> line.

Other evolutionary parameters can be changed accordingly to the specific problems, but for the benchmark problems they were fixed. 

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


## Citation
If you use this code, please cite our paper:

```bibtex
@inproceedingsdoz2025evolutionary,
  title={Evolutionary Synthesis of Probabilistic Programs},
  author={Doz, Romina and Randone, Francesca and Medvet, Eric and Bortolussi, Luca},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  year={2025}
}
```
## Reference

Michael Fenton, James McDermott, David Fagan, Stefan Forstenlechner, Erik
Hemberg, and Michael O’Neill. 2017. Ponyge2: Grammatical evolution in python.
In Proceedings of the Genetic and Evolutionary Computation Conference Companion.
1194–1201
=======
# Causal_Models_Synthesis
>>>>>>> e8ea544233829b1805c943201ade0fa58cb08bcd
