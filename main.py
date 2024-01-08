import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# Define constants
N = 40  # Size of the grid (N x N)
sigma = 1  # Selection rate
mu = 1  # Reproduction rate
epsilon = 15  # Movement rate
num_generations = 10000  # Number of generations to simulate

# Calculate total rate
total_rate = sigma + mu + epsilon
# Define a mapping from strings to numbers
str_to_num = {"": 0, "paper": 1, "scissors": 2, "rock": 3}

# Neue Variable für den aktuellen Frame hinzufügen
current_frame = 0
color_distribution = {"Empty": 0, "Paper": 0, "Scissors": 0, "Rock": 0}


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

# Initialisiere reproduction_count auf 0
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
img = ax.imshow(np.vectorize(str_to_num.get)(grid), cmap=cmap, norm=norm)
colorbar = plt.colorbar(img, cax=cax, ticks=[0.5, 1.5, 2.5, 3.5])
colorbar.ax.set_yticklabels(["Empty", "Paper", "Scissors", "Rock"])

# Anzeige der Gewichtungen links vom Grid
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

            # Determine the winner of RPS
            winner = rps_winner(individual, neighbor)

            # The loser dies, and its cell becomes empty
            if winner != individual:
                grid[x, y] = ""
            # else:
            # grid[neighbor_x, neighbor_y] = ""

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


# Erstelle die Animation
ani = FuncAnimation(fig, update, frames=num_generations, interval=50, repeat=False)


# Funktionen für die Buttons
def start_animation(event):
    ani.event_source.start()


def stop_animation(event):
    ani.event_source.stop()


# Funktion zum Zurücksetzen der Animation aktualisieren
def reset_animation(event):
    global grid
    global reproduction_count
    global current_frame
    grid = np.random.choice(["rock", "paper", "scissors"], size=(N, N))
    reproduction_count = 0
    current_frame = 0  # Setze den aktuellen Frame zurück
    img.set_array(np.vectorize(str_to_num.get)(grid))
    ax.set_title("Rock-Paper-Scissors Grid - Generation 0")
    ani.frame_seq = ani.new_frame_seq()  # Setze den Frame-Generator zurück
    ani.event_source.stop()


# Buttons hinzufügen
ax_start = plt.axes([0.1, 0.01, 0.1, 0.05])
ax_stop = plt.axes([0.25, 0.01, 0.1, 0.05])
ax_reset = plt.axes([0.4, 0.01, 0.1, 0.05])

btn_start = Button(ax_start, "Start")
btn_start.on_clicked(start_animation)

btn_stop = Button(ax_stop, "Stop")
btn_stop.on_clicked(stop_animation)

btn_reset = Button(ax_reset, "Reset")
btn_reset.on_clicked(reset_animation)


plt.show()
