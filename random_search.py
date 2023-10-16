from GA import *
import random

grid_num_iter = [500, 1000, 1500]
grid_num_bits = [20]
grid_num_pop = [50, 100, 150]
grid_cross_rate = [0.5, 0.7, 0.9]
grid_mut_rate = [0.01, 0.1, 0.5]
grid_mut_step = [0.001, 0.01, 0.1]
grid_crossover_method = ["spx", "pmx", "cx"]
grid_selection_method = ["tournament", "roulette"]

# Perform random search
best_fitness = float('inf')
best_hyperparameters = None
total_iterations = 100

for iteration in range(total_iterations):
    # Randomly choose hyperparameters from the grid
    num_iter = random.choice(grid_num_iter)
    num_pop = random.choice(grid_num_pop)
    cross_rate = random.choice(grid_cross_rate)
    mut_rate = random.choice(grid_mut_rate)
    mut_step = random.choice(grid_mut_step)
    crossover_method = random.choice(grid_crossover_method)
    selection_method = random.choice(grid_selection_method)

    print(f"Running random search with iteration {iteration + 1}...")
    _, best_num = GA(objective_1, num_bits, num_iter, num_pop, cross_rate, mut_rate, mut_step, crossover_method,
                     selection_method)

    if best_num < best_fitness:
        best_fitness = best_num
        best_hyperparameters = (num_iter, num_bits, num_pop, cross_rate, mut_rate, mut_step, crossover_method,
                                selection_method)

print("\nBest Hyperparameters:")
print("num_iter:", best_hyperparameters[0])
print("num_bits:", best_hyperparameters[1])
print("num_pop:", best_hyperparameters[2])
print("cross_rate:", best_hyperparameters[3])
print("mut_rate:", best_hyperparameters[4])
print("mut_step:", best_hyperparameters[5])
print("crossover_method:", best_hyperparameters[6])
print("selection_method:", best_hyperparameters[7])

# Run the GA again with the best hyperparameters and selection method and get the best individual and fitness
best, best_num = GA(objective_1, *best_hyperparameters[1:7], crossover_method=best_hyperparameters[6],
                    selection_method=best_hyperparameters[7])
print("\nBest Genes:", best)
print("Best Fitness:", best_num)