# Genetic Algorithm for Minimization of Two Objective Functions

## Project Overview
This project implements a Genetic Algorithm (GA) to minimize two distinct objective functions. A genetic algorithm is an optimization technique based on the principles of natural selection and genetics, which is useful for solving complex problems that are difficult to solve using traditional methods. For a more detailed explanation refer to the [report](./report/Assignment%20AI.docx).

The project focuses on minimizing the following two objective functions:

1. **Objective 1**: A quadratic-based function with dependencies on previous variables.
   
   ![First minimsation function](./imgs/min_func_1.png?raw=true)
   ![First minimsation function](./imgs/min_func_graph_1.png?raw=true)
   
2. **Objective 2**: A polynomial function of degree 4 with additional constraints.
   
   ![First minimsation function](./imgs/min_func_2.png?raw=true)
   ![First minimsation function](./imgs/min_func_graph_2.png?raw=true)
   ![First minimsation function](./imgs/min_func_2_ans.png?raw=true)

## Requirements
- Python 3.x
- NumPy
- Matplotlib

## Installation
You can install the required dependencies using the following command:

```bash
pip install numpy matplotlib
```

## Project Structure
The project consists of the following key components ([GA.py](./GA.py)):

### 1. Objective Functions
- **`objective_1(x)`**: A complex quadratic-based minimization function.
- **`objective_2(x)`**: A polynomial-based minimization function.

### 2. Genetic Algorithm Components
- **Selection Methods**:
  - **Tournament Selection**: Selects individuals from the population based on a tournament mechanism.
  - **Roulette Wheel Selection**: Selects individuals probabilistically based on their fitness.

- **Crossover Methods**:
  - **Single-Point Crossover (SPX)**: Swaps segments between two parents at a single crossover point.
  - **Partially Mapped Crossover (PMX)**: Preserves the order of genes and maps segments between two parents.
  - **Cycle Crossover (CX)**: Creates offspring by building cycles from both parents.

- **Mutation**:
  - Randomly perturbs individual genes to introduce variability and explore the search space.

### 3. Genetic Algorithm Function
- **`GA(objective_1, num_bits, num_iter, num_pop, cross_rate, mut_rate, mut_step, crossover_method, selection_method)`**: The core function that runs the genetic algorithm. It initializes a population, iteratively applies selection, crossover, and mutation, and tracks the best solutions over generations.

### 4. Visualization
- The results of the genetic algorithm are visualized using Matplotlib, showing the best and average fitness across generations.

## Usage
To run the genetic algorithm and minimize the first objective function:

```python
best, best_num = GA(
    objective_1, 
    num_bits=20, 
    num_iter=1000, 
    num_pop=100, 
    cross_rate=0.9, 
    mut_rate=1.0 / 20, 
    mut_step=0.5, 
    crossover_method="spx", 
    selection_method="tournament"
)
```

This will print the best set of genes and their corresponding fitness value. Additionally, a plot of the best and mean fitness values across generations will be displayed.

## Customization
- **Objective Function**: Change the objective function by passing `objective_2` to the `GA` function.
- **Crossover and Selection Methods**: Customize the algorithm by changing the `crossover_method` and `selection_method` parameters. Available methods include:
  - `crossover_method`: `"spx"`, `"pmx"`, `"cx"`
  - `selection_method`: `"tournament"`, `"roulette"`

## Example Output
```bash
Best Genes: [0.1, -0.2, ... , 1.5]
Best Fitness: 0.0003
```

The plot will show the convergence of the genetic algorithm over the generations.

## Conclusion
This project provides a flexible framework to apply genetic algorithms for minimizing complex objective functions. By adjusting the parameters, selection methods, and crossover techniques, you can explore various optimization scenarios and improve the performance of the algorithm.
