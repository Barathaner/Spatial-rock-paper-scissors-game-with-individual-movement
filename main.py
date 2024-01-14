import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# Define constants
N = 100  # Size of the grid (N x N)
sigma = 1  # Selection rate
mu = 1  # Reproduction rate
epsilon = 10  # Movement rate
num_generations = 10000  # Number of maximum generations to simulate

# Calculate total rate
total_rate = sigma + mu + epsilon
# Define a mapping from strings to numbers
str_to_num = {"": 0, "paper": 1, "scissors": 2, "rock": 3}

# Add a new variable for the current frame
current_frame = 0
color_distribution = {"Empty": 0, "Paper": 0, "Scissors": 0, "Rock": 0}


# Define Rock-Paper-Scissors rules
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

# Initialize reproduction_count to 0
reproduction_count = 0

# Create a color map for the states
cmap = mcolors.ListedColormap(["#FFFFFF", "#00FF00", "#0000FF", "#FF0000"])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Define neighbor coordinates (top, bottom, left, right)
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Create the plot and colorbar
fig, (ax, cax) = plt.subplots(
    1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1, 0.05]}
)

# Shift the Colorbar to the left
fig.subplots_adjust(right=0.8)

img = ax.imshow(np.vectorize(str_to_num.get)(grid), cmap=cmap, norm=norm)
colorbar = plt.colorbar(img, cax=cax, ticks=[0.5, 1.5, 2.5, 3.5])
colorbar.ax.set_yticklabels(["Empty", "Paper", "Scissors", "Rock"])

# Display the weights to the left of the grid
ax.text(
    -0.1,
    0.5,
    f"Sigma={sigma}\nMu={mu}\nEpsilon={epsilon}",
    transform=ax.transAxes,
    va="center",
    ha="right",
)

ax.set_title("Rock-Paper-Scissors Grid - Generation 0")


# Function to update the plot for each generation
def update(frame):
    # Calculate the percentage of each color at the beginning of the generation
    color_percentage = {
        color.capitalize(): np.count_nonzero(grid == color) / grid.size * 100
        for color in str_to_num.keys()
    }

    # Update Colorbar ticks with percentage values
    colorbar.ax.set_yticklabels(
        [
            f"Empty - {color_percentage['']:.2f}%",
            f"Paper - {color_percentage['Paper']:.2f}%",
            f"Scissors - {color_percentage['Scissors']:.2f}%",
            f"Rock - {color_percentage['Rock']:.2f}%",
        ]
    )

    # Check if all cells have the same color
    unique_colors = np.unique(grid)
    if len(unique_colors) == 2 and "" in unique_colors:
        # If only one non-empty color exists, stop the animation
        ani.event_source.stop()

        # Find the remaining color
        remaining_color = [color for color in unique_colors if color != ""][0]

        # Display a message on the plot
        ax.text(
            0.5,
            1.1,
            f"Only {remaining_color.capitalize()} remain",
            transform=ax.transAxes,
            va="center",
            ha="center",
            fontsize=12,
            color="red",
        )

        return

    # Reset reproduction_count for the next generation
    reproduction_count = 0

    while reproduction_count < N:
        # Choose a random event based on rates
        event = random.choices(
            ["selection", "reproduction", "movement"], weights=[sigma, mu, epsilon]
        )[0]

        # Choose a random individual
        x, y = random.randint(0, N - 1), random.randint(0, N - 1)
        individual = grid[x, y]

        if event == "selection":
            # Choose a random neighbor
            dx, dy = random.choice(neighbors)
            neighbor_x, neighbor_y = (x + dx) % N, (
                y + dy
            ) % N  # Apply periodic boundaries
            neighbor = grid[neighbor_x, neighbor_y]
            if neighbor == "":
                continue

            # Determine the winner of RPS
            winner = rps_winner(individual, neighbor)

            # The loser dies, and its cell becomes empty
            if winner != individual:
                grid[x, y] = ""

        elif event == "reproduction":
            # Check for empty neighboring cells
            empty_neighbors = [
                (x + dx, y + dy)
                for dx, dy in neighbors
                if grid[(x + dx) % N, (y + dy) % N] == ""
            ]
            if empty_neighbors:
                # Choose a random empty neighbor
                empty_x, empty_y = random.choice(empty_neighbors)
                grid[empty_x % N, empty_y % N] = individual

            # Increment the reproduction count for the current generation
            reproduction_count += 1

        elif event == "movement":
            # Choose a random neighbor (including empty cells)
            dx, dy = random.choice(neighbors)
            neighbor_x, neighbor_y = (x + dx) % N, (
                y + dy
            ) % N  # Apply periodic boundaries

            # Swap positions of the two individuals
            grid[x, y], grid[neighbor_x, neighbor_y] = (
                grid[neighbor_x, neighbor_y],
                grid[x, y],
            )

    # Update the plot with the current state of the grid
    img.set_array(np.vectorize(str_to_num.get)(grid))
    ax.set_title("Rock-Paper-Scissors Grid - Generation {}".format(frame))


# Create the animation
ani = FuncAnimation(fig, update, frames=num_generations, interval=50, repeat=False)


# Functions for the buttons
def start_animation(event):
    ani.event_source.start()


def stop_animation(event):
    ani.event_source.stop()


# Update reset animation function
def reset_animation(event):
    global grid
    global reproduction_count
    global current_frame
    grid = np.random.choice(["rock", "paper", "scissors"], size=(N, N))
    reproduction_count = 0
    current_frame = 0  # Reset the current frame
    img.set_array(np.vectorize(str_to_num.get)(grid))
    ax.set_title("Rock-Paper-Scissors Grid - Generation 0")
    ani.frame_seq = ani.new_frame_seq()  # Reset the frame generator
    ani.event_source.stop()


# Add buttons
ax_start = plt.axes([0.15, 0.01, 0.1, 0.05])
ax_stop = plt.axes([0.35, 0.01, 0.1, 0.05])
ax_reset = plt.axes([0.56, 0.01, 0.1, 0.05])

btn_start = Button(ax_start, "Start")
btn_start.on_clicked(start_animation)

btn_stop = Button(ax_stop, "Stop")
btn_stop.on_clicked(stop_animation)

btn_reset = Button(ax_reset, "Reset")
btn_reset.on_clicked(reset_animation)

plt.show()
