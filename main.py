import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# Define constants
N = 300  # Size of the grid (N x N)
num_generations = 5000  # Number of generations

# Initialize parameters (these will be controlled by sliders)
sigma = 1.0  # Selection rate
mu = 1.0  # Reproduction rate
epsilon = 0.00001  # Movement rate

# Initialize the grid with larger clusters
grid = np.zeros((N, N), dtype=int)
cluster_size = 20
for i in range(0, N, cluster_size):
    for j in range(0, N, cluster_size):
        grid[i:i+cluster_size, j:j+cluster_size] = np.random.randint(1, 4)

# Create a color map for the states
cmap = mcolors.ListedColormap(["#FFFFFF", "#FFFF00", "#0000FF", "#FF0000"])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Create the plot and colorbar
fig, (ax, cax) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 0.05]})
fig.subplots_adjust(right=0.8, bottom=0.25)
fig.canvas.manager.set_window_title('Spatial Rock Paper Scissors')

img = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
colorbar = plt.colorbar(img, cax=cax, ticks=[0.5, 1.5, 2.5, 3.5])
colorbar.ax.set_yticklabels(["Empty", "Paper", "Scissors", "Rock"])

ax.set_title("Rock-Paper-Scissors Grid - Generation 0")

# Precompute neighbor indices
neighbors = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

def update(frame):
    global grid
    
    # Calculate total rate
    total_rate = sigma + mu + epsilon
    
    new_grid = grid.copy()
    
    # Selection
    mask = np.random.random(size=(N, N)) < sigma / total_rate
    neighbor_coords = (np.indices((N, N)).T[:,:,None] + neighbors[np.random.randint(0, 4, size=(N, N))][:,:,None]).reshape(-1, 2) % N
    neighbor_values = grid[neighbor_coords[:,0], neighbor_coords[:,1]].reshape(N, N)
    new_grid[mask & ((grid % 3 + 1) == neighbor_values)] = 0
    
    # Reproduction
    mask = (np.random.random(size=(N, N)) < mu / total_rate) & (new_grid == 0)
    if mask.any():
        reproducing_coords = np.argwhere(mask)
        neighbor_coords = (reproducing_coords[:, None] + neighbors[np.random.randint(0, 4, size=len(reproducing_coords))][:, None]).reshape(-1, 2) % N
        neighbor_values = grid[neighbor_coords[:,0], neighbor_coords[:,1]]
        valid_reproductions = neighbor_values != 0
        new_grid[reproducing_coords[valid_reproductions,0], reproducing_coords[valid_reproductions,1]] = neighbor_values[valid_reproductions]
    
    # Movement
    mask = np.random.random(size=(N, N)) < epsilon / total_rate
    if mask.any():
        moving_coords = np.argwhere(mask)
        swap_coords = (moving_coords + neighbors[np.random.randint(0, 4, size=len(moving_coords))]) % N
        new_grid[moving_coords[:,0], moving_coords[:,1]], new_grid[swap_coords[:,0], swap_coords[:,1]] = \
            new_grid[swap_coords[:,0], swap_coords[:,1]], new_grid[moving_coords[:,0], moving_coords[:,1]]

    grid = new_grid

    # Update the plot with the current state of the grid
    img.set_array(grid)
    ax.set_title(f"Rock-Paper-Scissors Grid - Generation {frame}")

    # Calculate and update color percentages
    unique, counts = np.unique(grid, return_counts=True)
    percentages = dict(zip(unique, counts / grid.size * 100))
    colorbar.ax.set_yticklabels([
        f"Empty - {percentages.get(0, 0):.2f}%",
        f"Paper - {percentages.get(1, 0):.2f}%",
        f"Scissors - {percentages.get(2, 0):.2f}%",
        f"Rock - {percentages.get(3, 0):.2f}%",
    ])

# Create the animation
ani = FuncAnimation(fig, update, frames=num_generations, interval=50, repeat=False)

# Functions for the buttons
def start_animation(event):
    ani.event_source.start()

def stop_animation(event):
    ani.event_source.stop()

def reset_animation(event):
    global grid
    grid = np.zeros((N, N), dtype=int)
    cluster_size = 20
    for i in range(0, N, cluster_size):
        for j in range(0, N, cluster_size):
            grid[i:i+cluster_size, j:j+cluster_size] = np.random.randint(1, 4)
    img.set_array(grid)
    ax.set_title("Rock-Paper-Scissors Grid - Generation 0")
    ani.frame_seq = ani.new_frame_seq()  # Reset the frame generator
    ani.event_source.stop()

# Add buttons
ax_start = plt.axes([0.15, 0.05, 0.1, 0.04])
ax_stop = plt.axes([0.35, 0.05, 0.1, 0.04])
ax_reset = plt.axes([0.56, 0.05, 0.1, 0.04])

btn_start = Button(ax_start, "Start")
btn_start.on_clicked(start_animation)

btn_stop = Button(ax_stop, "Stop")
btn_stop.on_clicked(stop_animation)

btn_reset = Button(ax_reset, "Reset")
btn_reset.on_clicked(reset_animation)

# Add sliders
ax_sigma = plt.axes([0.15, 0.15, 0.5, 0.03])
ax_mu = plt.axes([0.15, 0.1, 0.5, 0.03])
ax_epsilon = plt.axes([0.15, 0.2, 0.5, 0.03])

slider_sigma = Slider(ax_sigma, 'Selection', -50, 50, valinit=sigma)
slider_mu = Slider(ax_mu, 'Reproduction', -50, 50, valinit=mu)
slider_epsilon = Slider(ax_epsilon, 'Movement', -50, 50, valinit=epsilon)

def update_params(val):
    global sigma, mu, epsilon
    sigma = slider_sigma.val
    mu = slider_mu.val
    epsilon = slider_epsilon.val

slider_sigma.on_changed(update_params)
slider_mu.on_changed(update_params)
slider_epsilon.on_changed(update_params)

plt.show()