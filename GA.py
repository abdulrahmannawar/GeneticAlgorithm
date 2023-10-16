import matplotlib.pyplot as plt
from numpy.random import randint, rand, uniform
import numpy as np
from pprint import pprint

num_iter = 1000
num_bits = 20
num_pop = 100
cross_rate = 0.9
mut_rate = 1.0 / float(num_bits)
mut_step = 0.5
MIN, MAX = -5, 5

best_ind = []
mean = []


# first minimization fitness function
def objective_1(x):
    sum = (x[0] - 1) ** 2
    for i in range(1, num_bits):
        sum += i * (2 * (x[i]) ** 2 - x[i - 1]) ** 2
    return sum


# second minimization fitness function
def objective_2(x):
    sum = 0
    for i in range(num_bits):
        sum += x[i] ** 4 - 16 * (x[i]) ** 2 + 5 * x[i]
    return sum * 0.5


# Tournament Selection Function
def tournament_selection(population, fitness, tournament_size):
    selected = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = [fitness[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitness)]
        selected.append(population[winner_index])
    return selected

def roulette_wheel_selection(population, fitness):
    positive_fitness_indices = [i for i, fit in enumerate(fitness) if fit > 0]
    if len(positive_fitness_indices) == 0:
        probabilities = [1 / len(population) for _ in range(len(population))]
    else:
        total_fitness = sum(fitness[i] for i in positive_fitness_indices)
        probabilities = [fitness[i] / total_fitness for i in positive_fitness_indices]

    selected_indices = np.random.choice(positive_fitness_indices, size=len(population), p=probabilities)
    selected_pop = [population[i] for i in selected_indices]
    return selected_pop

def selection(population, fitness, selection_method="tournament"):
    if selection_method == "tournament":
        return tournament_selection(population, fitness, tournament_size=3)
    elif selection_method == "roulette":
        return roulette_wheel_selection(population, fitness)
    else:
        raise ValueError("Invalid selection method")

def crossover(parent_1, parent_2, cross_rate, crossover_method="spx"):
    if crossover_method == "spx":
        return sp_crossover(parent_1, parent_2, cross_rate)
    elif crossover_method == "pmx":
        return pmx_crossover(parent_1, parent_2, cross_rate)
    elif crossover_method == "cx":
        return cx_crossover(parent_1, parent_2, cross_rate)
    else:
        raise ValueError("Invalid crossover method")


def sp_crossover(parent_1, parent_2, cross_rate):
    # making children copies of parents in case crossover doesn't occur
    # check for recombination probability
    if rand() < cross_rate:
        # select crossover point that is not on the end of the string
        cross_point = randint(1, len(parent_1) - 2)
        # placing split point from number generated from crosspoint
        child_1 = parent_1[:cross_point] + parent_2[cross_point:]
        child_2 = parent_2[:cross_point] + parent_1[cross_point:]

        for i in range(len(child_1)):
            if child_1[i] > MAX:
                child_1[i] = MAX
            if child_1[i] < MIN:
                child_1[i] = MIN
            if child_2[i] > MAX:
                child_2[i] = MAX
            if child_2[i] < MIN:
                child_2[i] = MIN

    else:
        child_1, child_2 = parent_1.copy(), parent_2.copy()

    return [child_1, child_2]

def pmx_crossover(parent_1, parent_2, cross_rate, max_iterations=100):
    if rand() < cross_rate:
        cross_point1 = randint(1, len(parent_1) - 2)
        cross_point2 = randint(cross_point1 + 1, len(parent_1) - 1)

        child_1 = [-1] * len(parent_1)
        child_2 = [-1] * len(parent_2)

        # Copy the middle segment from parents to children
        child_1[cross_point1:cross_point2] = parent_1[cross_point1:cross_point2]
        child_2[cross_point1:cross_point2] = parent_2[cross_point1:cross_point2]

        # Fill the remaining genes using PMX
        iteration = 0
        while iteration < max_iterations:
            for i in range(cross_point1, cross_point2):
                if child_1[i] == -1:
                    if parent_1[i] not in child_2:
                        child_1[i] = parent_1[i]
                    else:
                        idx = parent_2.index(parent_1[i])
                        while child_1[idx] != -1:
                            idx = parent_2.index(parent_1[idx])
                        child_1[idx] = parent_1[i]

                if child_2[i] == -1:
                    if parent_2[i] not in child_1:
                        child_2[i] = parent_2[i]
                    else:
                        idx = parent_1.index(parent_2[i])
                        while child_2[idx] != -1:
                            idx = parent_1.index(parent_2[idx])
                        child_2[idx] = parent_2[i]

            iteration += 1

            # Check if all genes are filled
            if -1 not in child_1 and -1 not in child_2:
                break

        # Fill the remaining genes with the mapped values
        for i in range(len(parent_1)):
            if child_1[i] == -1:
                child_1[i] = parent_2[i]
            if child_2[i] == -1:
                child_2[i] = parent_1[i]

        return [child_1, child_2]

    return [parent_1, parent_2]


def cx_crossover(parent_1, parent_2, cross_rate):
    if rand() < cross_rate:
        child_1 = [-1] * len(parent_1)
        child_2 = [-1] * len(parent_2)

        cycle_start = randint(len(parent_1))
        idx = cycle_start

        # Create a cycle
        while True:
            child_1[idx] = parent_1[idx]
            child_2[idx] = parent_2[idx]
            if parent_1[idx] == parent_2[cycle_start]:
                break
            idx = parent_1.index(parent_2[idx]) if parent_2[idx] in parent_1 else None
            if idx is None:
                break

        # Fill the remaining genes using CX
        for i in range(len(parent_1)):
            if child_1[i] == -1:
                child_1[i] = parent_2[i]
            if child_2[i] == -1:
                child_2[i] = parent_1[i]

        return [child_1, child_2]

    return [parent_1, parent_2]


def mutation(individual, mut_rate, mut_step, MIN, MAX):
    if isinstance(individual, list):
        # Mutation for a list of genes
        for i in range(len(individual)):
            # Check for mutation probability
            if rand() < mut_rate:
                mutate = uniform(-mut_step, mut_step)  # Select how much the gene should mutate by
                individual[i] += mutate
                # Check that gene is not out of bounds
                if individual[i] > MAX:
                    individual[i] = MAX
                if individual[i] < MIN:
                    individual[i] = MIN
    else:
        if rand() < mut_rate:
            mutate = uniform(-mut_step, mut_step)  # Select how much the gene should mutate by
            individual += mutate
            # Check that gene is not out of bounds
            if individual > MAX:
                individual = MAX
            if individual < MIN:
                individual = MIN


# main GA function
def GA(objective_1, num_bits, num_iter, num_pop, cross_rate, mut_rate, mut_step, crossover_method, selection_method):
    population = []
    for i in range(num_pop):
        individual = [uniform(MIN, MAX) for _ in range(num_bits)]
        population.append(individual)  # Initialize pop with individuals filled with genes
    # Keep track of the best solution and run the objective function on the first index to use as a comparison
    best, best_num = population[0], objective_1(population[0])
    best_ind.append((best, best_num))  # Append the initial best individual to the list
    # Enumerate over multiple generations
    for _ in range(num_iter):
        # Evaluate candidates in the pop by using the objective func on each individual
        fitness = [objective_1(individual) for individual in population]
        mean.append(sum(fitness) / num_pop)  # Append the average fitness of every individual to plot on the graph
        # Check for a new best solution continuously
        for i in range(num_pop):
            if fitness[i] < best_num:
                best, best_num = population[i], fitness[i]
                best_ind.append((best, best_num))  # Getting the best individuals' genes and fitness and appending to a list to plot on the graph
        # Select parents using Tournament Selection
        selected = selection(population, fitness, selection_method=selection_method)
        # Create the next generation of children
        children = list()
        for i in range(0, num_pop, 2):
            # Get selected parents in pairs
            parent_1, parent_2 = selected[i], selected[i + 1]
            # Crossover & mutation (explained above)
            for c in crossover(parent_1, parent_2, cross_rate, crossover_method=crossover_method):
                # Mutation
                if isinstance(c, list):
                    mutation(c, mut_rate, mut_step, MIN, MAX)
                else:
                    mutation(c, mut_rate, mut_step, MIN, MAX)
                # Store for the next gen to run the func on again to get an even better result
                children.append(c)
        # Replace population
        population = children
    return (best, best_num)  # Returns the best individual with the best fitness


best, best_num = GA(objective_1, num_bits, num_iter, num_pop, cross_rate, mut_rate, mut_step, crossover_method="spx", selection_method="tournament")
pprint(f"Best Genes: {best}")
print("Best Fitness:", best_num)

# Plotting lists of mean and best individuals to plot on the graph
best_genes = [individual for individual, _ in best_ind]
best_fitness = [fitness for _, fitness in best_ind]

plt.plot(best_fitness, label="best")
plt.plot(mean, label="mean")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()
