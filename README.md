# Spatial Rock-Paper-Scissors Game with Individual Movement
<img width="983" alt="rps" src="https://github.com/user-attachments/assets/b02aa7cb-b378-410a-afd7-c407c283d3ac">

Welcome to the **Spatial Rock-Paper-Scissors Game with Individual Movement**! This project is a small-scale implementation inspired by the paper "Mobility promotes and jeopardizes biodiversity in rock-paper-scissors games" by Tobias Reichenbach, Mauro Mobilia, and Erwin Frey. This was a project for the course Computational Biology and Mathematics at Université Aix-Marseille.

## Overview

In this simulation, we explore how individual movement impacts biodiversity using a spatial grid representation of the classic rock-paper-scissors game. The simulation allows you to adjust key parameters such as selection, reproduction, and movement rates, providing insight into their effects on population dynamics.

Check out my YouTube video explaining this project: [Watch Video](https://www.youtube.com/watch?v=5jrQNysCAkU)

## Features

- **NxN Grid**: The simulation operates on a configurable NxN grid where each cell represents an individual of a species or an empty space.
- **Adjustable Parameters**: Use sliders to tweak the selection, reproduction, and movement rates during the simulation.
- **Real-Time Visualization**: Watch the evolution of populations and observe how different rates affect the dynamics in real-time.
- **Interactive**: Modify parameters on the fly and see the immediate impact on the simulation.

## Parameters

- **Selection Rate**: Controls how often individuals compete with neighbors.
- **Reproduction Rate**: Determines how frequently individuals reproduce.
- **Movement Rate**: Governs the likelihood of individuals moving to neighboring cells.

The rates are normalized as follows:

\[ \text{Mobility} = \frac{\text{mobility rate}}{\text{selection rate} + \text{reproduction rate} + \text{mobility rate}} \]

## Installation and Usage

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/spatial-rock-paper-scissors-game.git
   cd spatial-rock-paper-scissors-game
   ```

2. **Install Dependencies**:
   Make sure you have `moviepy` and other required libraries installed. You can install them using pip:
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Simulation**:
   ```sh
   python main.py
   ```

4. **Interact with the Simulation**:
   Adjust the sliders to change the selection, reproduction, and movement rates and observe how the populations evolve over generations.

## Contributing

Contributions are welcome! Feel free to debug, add features, or make improvements. Here’s how you can contribute:

1. **Fork the Repository**: Click on the "Fork" button on the top right corner of this page.
2. **Clone Your Fork**: Clone your forked repository to your local machine.
   ```sh
   git clone https://github.com/yourusername/spatial-rock-paper-scissors-game.git
   ```
3. **Create a Branch**: Create a new branch for your feature or bugfix.
   ```sh
   git checkout -b feature/your-feature-name
   ```
4. **Commit Your Changes**: Make your changes and commit them with a meaningful commit message.
   ```sh
   git commit -m "Add feature: your feature description"
   ```
5. **Push to Your Fork**: Push your changes to your forked repository.
   ```sh
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request**: Go to the original repository and create a pull request from your fork.

## Issues

If you encounter any issues, please create an issue on the repository’s GitHub page. Describe the problem you’re facing in detail, and we’ll do our best to help you out.

## Acknowledgements

This project is based on the research by Tobias Reichenbach, Mauro Mobilia, and Erwin Frey. Special thanks to the professors and students of the Computational Biology and Mathematics course at Université Aix-Marseille for their support and guidance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Have Fun!

Dive into the simulation, experiment with different parameters, and observe the fascinating dynamics of populations in a spatial rock-paper-scissors game. Enjoy and happy simulating!

---

Feel free to modify this README as needed to better fit your project and its requirements.
