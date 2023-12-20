import numpy as np
import random

# Define constants
N = 10  # Size of the grid (N x N)
sigma = 0.1  # Selection rate
mu = 0.2     # Reproduction rate
epsilon = 0.3  # Movement rate
num_generations = 100  # Number of generations to simulate

# Define RPS rules
def rps_winner(a, b):
    if a == "rock" and b == "scissors":
        return "rock"
    elif a == "scissors" and b == "paper":
        return "scissors"
    elif a == "paper" and b == "rock":
        return "paper"
    else:
        return b

# Initialize the grid with random individuals
grid = np.random.choice(["rock", "paper", "scissors"], size=(N, N))

# Define neighbor coordinates (top, bottom, left, right)
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Simulation loop
for generation in range(num_generations):
    # Calculate total rate
    total_rate = sigma + mu + epsilon
    
    # Calculate time until the next event
    time_until_event = random.expovariate(total_rate)
    
    # Choose a random event based on rates
    event = random.choices(["selection", "reproduction", "movement"],
                           weights=[sigma, mu, epsilon])[0]
    
    # Choose a random individual
    x, y = random.randint(0, N-1), random.randint(0, N-1)
    individual = grid[x, y]
    
    if event == "selection":
        # Choose a random neighbor
        dx, dy = random.choice(neighbors)
        neighbor_x, neighbor_y = (x + dx) % N, (y + dy) % N  # Apply periodic boundaries
        neighbor = grid[neighbor_x, neighbor_y]
        
        # Determine the winner of RPS
        winner = rps_winner(individual, neighbor)
        
        # The loser dies, and its cell becomes empty
        if winner != individual:
            grid[x, y] = ""
    
    elif event == "reproduction":
        # Check for empty neighboring cells
        empty_neighbors = [(x + dx, y + dy) for dx, dy in neighbors if grid[(x + dx) % N, (y + dy) % N] == ""]
        
        if empty_neighbors:
            # Choose a random empty neighbor
            empty_x, empty_y = random.choice(empty_neighbors)
            grid[empty_x % N, empty_y % N] = individual
    
    elif event == "movement":
        # Choose a random neighbor (including empty cells)
        dx, dy = random.choice(neighbors)
        neighbor_x, neighbor_y = (x + dx) % N, (y + dy) % N  # Apply periodic boundaries
        
        # Swap positions of the two individuals
        grid[x, y], grid[neighbor_x, neighbor_y] = grid[neighbor_x, neighbor_y], grid[x, y]

# Visualization: You can use matplotlib to create heat plots and visualize the results.
