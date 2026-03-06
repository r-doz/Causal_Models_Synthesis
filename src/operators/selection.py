from random import sample
import random
from collections import defaultdict
from typing import List

from algorithm.parameters import params
from utilities.algorithm.NSGA2 import compute_pareto_metrics, \
    crowded_comparison_operator


def selection(population):
    """
    Perform selection on a population in order to select a population of
    individuals for variation.

    :param population: input population
    :return: selected population
    """

    return params['SELECTION'](population)


def tournament(population):
    """
    Given an entire population, draw <tournament_size> competitors randomly and
    return the best. Only valid individuals can be selected for tournaments.

    :param population: A population from which to select individuals.
    :return: A population of the winners from tournaments.
    """

    # Initialise list of tournament winners.
    winners = []

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    while len(winners) < params['GENERATION_SIZE']:
        # Randomly choose TOURNAMENT_SIZE competitors from the given
        # population. Allows for re-sampling of individuals.
        competitors = sample(available, params['TOURNAMENT_SIZE'])

        # Return the single best competitor.
        winners.append(max(competitors))

    # Return the population of tournament winners.
    return winners


def truncation(population):
    """
    Given an entire population, return the best <proportion> of them.

    :param population: A population from which to select individuals.
    :return: The best <proportion> of the given population.
    """

    # Sort the original population.
    population.sort(reverse=True)

    # Find the cutoff point for truncation.
    cutoff = int(len(population) * float(params['SELECTION_PROPORTION']))

    # Return the best <proportion> of the given population.
    return population[:cutoff]


def nsga2_selection(population):
    """Apply NSGA-II selection operator on the *population*. Usually, the
    size of *population* will be larger than *k* because any individual
    present in *population* will appear in the returned list at most once.
    Having the size of *population* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *population*. For more
    details on the NSGA-II operator see [Deb2002]_.
    
    :param population: A population from which to select individuals.
    :returns: A list of selected individuals.
    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """

    selection_size = params['GENERATION_SIZE']
    tournament_size = params['TOURNAMENT_SIZE']

    # Initialise list of tournament winners.
    winners = []

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    # Compute pareto front metrics.
    pareto = compute_pareto_metrics(available)

    while len(winners) < selection_size:
        # Return the single best competitor.
        winners.append(pareto_tournament(available, pareto, tournament_size))

    return winners


def pareto_tournament(population, pareto, tournament_size):
    """
    The Pareto tournament selection uses both the pareto front of the
    individual and the crowding distance.

    :param population: A population from which to select individuals.
    :param pareto: The pareto front information.
    :param tournament_size: The size of the tournament.
    :return: The selected individuals.
    """

    # Initialise no best solution.
    best = None

    # Randomly sample *tournament_size* participants.
    participants = sample(population, tournament_size)

    for participant in participants:
        if best is None or crowded_comparison_operator(participant, best,
                                                       pareto):
            best = participant

    return best


# Set attributes for all operators to define multi-objective operators.
nsga2_selection.multi_objective = True



def stratified_selection(population):
    """
    Stratified (Class-Conditional) Selection.

    Two-stage selection:
        1) Sample a class (structure_id)
        2) Perform tournament selection within that class
    """
    tournament_size = params['TOURNAMENT_SIZE']
    selection_size = params['GENERATION_SIZE']

    winners = []
    # ------------------------------------------------------------
    # Step 0: filter valid individuals
    # ------------------------------------------------------------
    available = [i for i in population if not i.invalid]

    if not available:
        raise ValueError("No valid individuals available for selection.")

    # ------------------------------------------------------------
    # Step 1: group individuals by class (structure_id)
    # ------------------------------------------------------------
    class_groups = defaultdict(list)
    for ind in available:
        class_groups[ind.structure_id].append(ind)

    classes = list(class_groups.keys())

    for _ in range(selection_size):
        # ------------------------------------------------------------
        # Step 2: sample a class (Stage A)
        # ------------------------------------------------------------
        chosen_class = random.choice(classes)
        # ------------------------------------------------------------
        # Step 3: tournament selection within chosen class (Stage B)
        # ------------------------------------------------------------
        candidates = class_groups[chosen_class]

        # If class is small, reduce tournament size
        k = min(tournament_size, len(candidates))

        tournament = random.sample(candidates, k)

        # Max fitness wins (assuming maximization)
        winner = max(tournament, key=lambda ind: ind.fitness)

        winners.append(winner)

    return winners
