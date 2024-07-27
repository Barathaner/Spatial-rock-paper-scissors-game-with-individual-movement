import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

# Set the plot style to dark background
plt.style.use('dark_background')

# Define constants
N = 100  # Size of the grid (N x N)
num_generations = 10000  # Number of generations

# Initialize parameters with default values
default_sigma = 1.0  # Default selection rate
default_mu = 1.0  # Default reproduction rate
default_epsilon = 1.0  # Default movement rate
min_rate = 0.0  # Minimum rate to prevent division by zero

sigma = default_sigma
mu = default_mu
epsilon = default_epsilon

# Initialize the grid with larger clusters
grid = np.random.randint(0, 4, size=(N, N))

# Create a color map for the states
cmap = mcolors.ListedColormap(["#1A1A1A", "#FFD700", "#00BFFF", "#FF4500"])
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Create the plot without the colorbar
fig, (ax, ax_pop) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1]})
fig.subplots_adjust(right=0.85, wspace=0.3, bottom=0.25)
fig.canvas.manager.set_window_title('Spatial Rock Paper Scissors - Dark Mode')

img = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')

ax.set_title("Rock-Paper-Scissors Grid - Generation 0", color='white')

# Initialize lists to store population counts
empty_count = []
paper_count = []
scissors_count = []
rock_count = []

# Precompute neighbor indices
neighbors = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

def update(frame):
    global grid, empty_count, paper_count, scissors_count, rock_count
    
    # Calculate total rate, ensuring it's never zero
    total_rate = max(sigma + mu + epsilon, min_rate)
    
    new_grid = grid.copy()
    
    # Selection
    mask = np.random.random(size=(N, N)) < sigma / total_rate
    neighbor_indices = np.random.randint(0, 4, size=(N, N))
    neighbor_coords = (np.indices((N, N)).T[:,:,None] + neighbors[neighbor_indices][:,:,None]).reshape(-1, 2) % N
    neighbor_values = grid[neighbor_coords[:,0], neighbor_coords[:,1]].reshape(N, N)
    new_grid[mask & ((grid % 3 + 1) == neighbor_values)] = 0
    
    # Reproduction
    mask = (np.random.random(size=(N, N)) < mu / total_rate) & (new_grid == 0)
    if mask.any():
        reproducing_coords = np.argwhere(mask)
        neighbor_indices = np.random.randint(0, 4, size=len(reproducing_coords))
        neighbor_coords = (reproducing_coords + neighbors[neighbor_indices]) % N
        neighbor_values = grid[neighbor_coords[:,0], neighbor_coords[:,1]]
        valid_reproductions = neighbor_values != 0
        new_grid[reproducing_coords[valid_reproductions,0], reproducing_coords[valid_reproductions,1]] = neighbor_values[valid_reproductions]
    
    # Movement
    mask = np.random.random(size=(N, N)) < epsilon / total_rate
    if mask.any():
        moving_coords = np.argwhere(mask)
        swap_indices = np.random.randint(0, 4, size=len(moving_coords))
        swap_coords = (moving_coords + neighbors[swap_indices]) % N
        new_grid[moving_coords[:,0], moving_coords[:,1]], new_grid[swap_coords[:,0], swap_coords[:,1]] = \
            new_grid[swap_coords[:,0], swap_coords[:,1]], new_grid[moving_coords[:,0], moving_coords[:,1]]

    grid = new_grid

    # Update the plot with the current state of the grid
    img.set_array(grid)
    ax.set_title(f"Rock-Paper-Scissors Grid - Generation {frame}", color='white')

    # Calculate and update color percentages
    unique, counts = np.unique(grid, return_counts=True)
    percentages = dict(zip(unique, counts / grid.size * 100))
    
    # Update population counts
    empty_count.append(percentages.get(0, 0))
    paper_count.append(percentages.get(1, 0))
    scissors_count.append(percentages.get(2, 0))
    rock_count.append(percentages.get(3, 0))

    # Update population plot
    ax_pop.clear()
    ax_pop.plot(empty_count, label="Empty", color="#1A1A1A")
    ax_pop.plot(paper_count, label="Paper", color="#FFD700")
    ax_pop.plot(scissors_count, label="Scissors", color="#00BFFF")
    ax_pop.plot(rock_count, label="Rock", color="#FF4500")
    ax_pop.set_title("Population Dynamics Over Time", color='white')
    ax_pop.set_xlabel("Generation", color='white')
    ax_pop.set_ylabel("Percentage", color='white')
    ax_pop.legend()
    ax_pop.grid(True)
    ax_pop.tick_params(colors='white')

# Create the animation (but don't start it yet)
ani = None

def init_animation():
    global ani
    ani = FuncAnimation(fig, update, frames=num_generations, interval=10, repeat=False)
    ani.event_source.stop()  # Make sure it's stopped initially

# Functions for the buttons
def start_animation(event):
    global ani
    if ani is None:
        init_animation()
    ani.event_source.start()

def stop_animation(event):
    global ani
    if ani is not None:
        ani.event_source.stop()

def reset_animation(event):
    global grid, sigma, mu, epsilon, ani, empty_count, paper_count, scissors_count, rock_count
    # Reset grid
    grid = np.random.randint(0, 4, size=(N, N))
    
    # Reset parameters to default values
    sigma = default_sigma
    mu = default_mu
    epsilon = default_epsilon
    
    # Update sliders
    slider_sigma.set_val(sigma)
    slider_mu.set_val(mu)
    slider_epsilon.set_val(epsilon)
    
    # Reset population counts
    empty_count = []
    paper_count = []
    scissors_count = []
    rock_count = []
    
    # Update plot
    img.set_array(grid)
    ax.set_title("Rock-Paper-Scissors Grid - Generation 0", color='white')
    
    # Reset animation
    if ani is not None:
        ani.frame_seq = ani.new_frame_seq()  # Reset the frame generator
        ani.event_source.stop()
    
    # Force a redraw of the plot
    fig.canvas.draw_idle()

# Add buttons
ax_start = plt.axes([0.15, 0.05, 0.1, 0.04])
ax_stop = plt.axes([0.35, 0.05, 0.1, 0.04])
ax_reset = plt.axes([0.56, 0.05, 0.1, 0.04])

btn_start = Button(ax_start, "Start", color='#4CAF50', hovercolor='#45a049')
btn_start.on_clicked(start_animation)

btn_stop = Button(ax_stop, "Stop", color='#f44336', hovercolor='#da190b')
btn_stop.on_clicked(stop_animation)

btn_reset = Button(ax_reset, "Reset", color='#008CBA', hovercolor='#007B9A')
btn_reset.on_clicked(reset_animation)

# Add sliders
ax_sigma = plt.axes([0.15, 0.15, 0.5, 0.03])
ax_mu = plt.axes([0.15, 0.1, 0.5, 0.03])
ax_epsilon = plt.axes([0.15, 0.2, 0.5, 0.03])

slider_sigma = Slider(ax_sigma, 'Selection', 0.000, 1.0, valinit=default_sigma, color='#FFA500')
slider_mu = Slider(ax_mu, 'Reproduction', 0.000, 1.0, valinit=default_mu, color='#FFA500')
slider_epsilon = Slider(ax_epsilon, 'Movement', 0.000, 1.00, valinit=default_epsilon, color='#FFA500')

def update_params(val):
    global sigma, mu, epsilon
    sigma = max(slider_sigma.val, min_rate)
    mu = max(slider_mu.val, min_rate)
    epsilon = max(slider_epsilon.val, min_rate)

slider_sigma.on_changed(update_params)
slider_mu.on_changed(update_params)
slider_epsilon.on_changed(update_params)

plt.show()
