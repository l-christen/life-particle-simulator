# User Guide : Life Particle Simulator
This user guide provides instructions on how to use the Life Particle Simulator, a GPU-based particle simulation integrated into Godot via GDExtension. It is intended for users who want to create and manage particle simulations within the Godot engine.

## Hardware Requirements
- A computer with a CUDA-capable GPU (NVIDIA).
- With the actual build settings, a computer with a Linux OS is required. (Windows and MacOS might work with some adjustments in the SConstruct file).

## Simulator access
You can access the simulator by cloning this repository and building the project using SCons or an executable file is provided in the releases section.

### Building from Source
Requirements:
* Linux OS :
    * GCC/G++
    * NVCC
    * SCons build system
    * Godot Engine (for the GUI)
* Windows OS (not tested) :
    * MSVC
    * NVCC
    * SCons build system
    * Godot Engine (for the GUI)

Note : You might need to adjust your C++ compiler version to match the NVCC supported version.


1. Clone the repository :
   ```bash
   git clone https://github.com/l-christen/life-particle-simulator.git
   cd life-particle-simulator
    ```
2. Retrieve Godot-cpp submodule :
    ```bash
    git submodule init
    git submodule update
    ```
3. Build Godot-cpp dynamic libraries :
    ```bash
    cd native/godot-cpp
    scons platform="linux/windows" target="template_debug/template_release"
    cd ..
    ```
4. Build the GDExtension dynamic library :
    ```bash
    scons platform="linux/windows" target="template_debug/template_release"
    ```
5. Open Godot Engine and load the project located in `./godot`.
6. Run the project or export it as an executable.

### Using Pre-built Executable (Linux only)
1. Go to the [releases]()
2. Download the latest release for Linux.
3. Extract the downloaded file. The executable need to be in the same folder as the .pck file.
4. Run the executable.

## Simulation parameters
- Number of particles per color (red, blue, green, yellow), max 250'000'000 particles per color.
- Interaction rules between colors (attraction, repulsion, neutral), sliders from -100 to 100. A negative value means repulsion, a positive value means attraction.
- Radius of interaction between particles, from 0 to 100'000. If you set a radius for a color, particles of that color will only interact with particles within that radius.
- Delta time for the simulation, from 0.0 to ???. A higher delta time means faster simulation, but can lead to instability.
- Viscosity of the environment, from 0.0 to ???. Behavior needs to be refined.

## Simulation physics
The simulation uses a simple particle interaction model based on attraction and repulsion forces. Each particle exerts a force on other particles based on the defined interaction rules and distance.

The force's calculation is inspired by the universal law of gravitation of Newton, but adapted for attraction and repulsion between different particle types.

Newton's formula for gravitational force:
$F = G \cdot (m_1 * m_2) / d^2$

In our simulation, we assume all particles have a unit mass (m_1 = m_2 = 1) and we replace the gravitational constant G with an interaction strength k defined by the user.

Formula for force calculation:
- The distance squared between two particles is calculated as : $d^2 = (x_2 - x_1)^2 + (y_2 - y_1)^2$
- If the distance squared is less than the interaction radius squared, a force is applied.
- The force magnitude is calculated as : $F = k / d^2$ where k is the interaction strength (positive for attraction, negative for repulsion).
- To avoid trigonometric functions, we get the normalized direction vector from particle 1 to particle 2 to determine x and y Force components, we calculate : $dir = ((x_2 - x_1) / d, (y_2 - y_1) / d)$
- The force vector applied to particle 1 is then : $F_{vector} = F * dir$
- A force is a mass times acceleration (F = m * a). Since we assume unit mass, the acceleration is equal to the force : $a_{vector} = F_{vector}$
- The particle's velocity is updated based on the acceleration and the delta time : $v_{new} = v_{old} + a_{vector} * deltaTime$
- TODO : REFINED THE WAY VISCOSITY IS APPLIED


## Once the simulator is running
The simulation/simulator can have 3 states : Stopped, Running and Paused.
- Stopped : No simulation is running. You can set up a new simulation.
- Running : The simulation is running in real-time.
- Paused : The simulation is paused.

### Stopped state
- In this state, you can modify all simulation parameters.
- Once you have set up the parameters, click on the "Start" button to start the simulation.

Add img

### Running state
- In this state, the simulation is running in real-time.
- You can click on the "Pause" button to pause the simulation.
- You can also click on the "Stop" button to stop the simulation and return to the Stopped state.

Add img

### Paused state
- In this state, the simulation is paused.
- You can modify all simulation parameters except the number of particles.
- You can click on the "Play" button to resume the simulation.

Add img
