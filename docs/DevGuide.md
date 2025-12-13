# Developer Guide: GPU Particle Simulation with CUDA and GDExtension

This developer guide provides an in-depth overview of the architecture and implementation details of the GPU-based particle simulation using CUDA, integrated into Godot via GDExtension. It is intended for developers looking to understand the code base.

---

## Architecture Overview

The implementation is organized into four distinct layers:

1. **CUDA (Simulation & Device Logic)**
2. **C++ Wrapper (Host - Device Interface)**
3. **GDExtension (Godot - C++ Bridge)**
4. **Godot (Scene, UI, Scripts)**

This separation ensures clarity, maintainability, and a clean boundary between simulation, native code, and engine-level integration.

---

## 1. CUDA Layer

This layer contains all GPU-side logic and memory management related to the particle simulation.

* **`native/src/simulation.cu`**

  * Core simulation logic
  * CUDA kernel implementation
  * Simulation initialization
  * Simulation step execution

* **`native/src/compute_buffers.cpp`**

  * CUDA memory allocation
  * Declaration of device buffers used for computation and rendering

* **`native/include/`**

  * CUDA data structures defining:

    * **SoA (Structure of Arrays)** for efficient GPU computation
    * **AoS (Array of Structures)** for efficient data transfer and rendering

---

## 2. C++ Wrapper Layer

This layer exposes a C++ API to interact with the CUDA simulation from the GDExtension.

* **`native/src/simulation.cuh`**

  * Declarations of `extern "C"` wrapper functions
  * Public interface used by the GDExtension to control the simulation

* **`native/src/simulation.cu`**

  * Implementation of the wrapper functions
  * Kernel launches, synchronization, and data transfers

* **`native/src/compute_buffers.cpp` / `native/include/compute_buffers.h`**

  * C++ classes encapsulating CUDA buffers
  * Abstraction layer between CUDA memory and C++ code

---

## 3. GDExtension Layer (C++)

This layer bridges the native C++ / CUDA code with Godot.

* **`native/src/cuda_particles_renderer.cpp`**
* **`native/include/cuda_particles_renderer.h`**

Responsibilities:
* Exposure of simulation controls to Godot
* Display particles by instantiating a MultiMeshInstance2D node
* High-level API used by GDScript to:

  * Initialize the simulation
  * Run simulation steps
  * Update simulation parameters during runtime

---

## 4. Godot Layer

This layer contains all engine-side logic.

* **`godot/`**

  * Godot scenes and nodes for visualization and interaction

* **`godot/scripts/`**

  * Camera control
  * `ControlScript.gd` acts as the main entry point for:

    * Calling GDExtension methods
    * Setting up the UI
    * Driving the simulation from the UI

* **`godot/extensions/`**

  * GDExtension configuration file
  * References the compiled native module (`.dll` / `.so`)

---
## Data structures and transfer
Computation data are stored in a C++ class called `ComputeBuffers`, which manages CUDA device memory for simulation and host memory for data transfer.

Data structures used:
* **SoA (Structure of Arrays)**: Optimized for GPU computation, used within CUDA kernels. For example, when we compute distances between particles, we only need positions, so having them in separate arrays allows coalesced memory access and not reading unnecessary data like type.
* **AoS (Array of Structures)**: Optimized for data acces CPU-side since the rendering needs the whole particle structure at once.

ComputeBuffers contains two SoA buffers, one for the previous frame and one for the current frame, avoiding read-write conflicts during simulation steps. It also contains one AoS buffer which contains the data to be sent to Godot for rendering.

A particle ready to be rendered is defined by an x and y position (float) and a type (uint32_t). The type defines the color of the particle.

A particle used for simulation contains velocity (float) and the type is used to define interaction rules between particles.

---

## Build System and Dependencies

* **`native/SConstruct`**

  * Build configuration for the GDExtension (Linux and Windows)
  * Handles compilation and linking of CUDA and C++ code

* **`godot-cpp` submodule**

  * Added as a dependency
  * Allows building the GDExtension without compiling the entire Godot engine

---
## Future Improvements
* Performance optimizations :
    * Full GPU to avoid data transfer each frame
    * Decorrelate simulation and rendering rates
    * Spatial partitioning

* Performance optimizations without full GPU : 
    * Use mapped memory to transfer data during computation, not after
    * Push positions and colors particles to Godot's MultiMesh in one call
* Simulation improvements :
    * Allow user to add more particles types
    * Add more interaction rules between particles
    * Allow user to record and playback simulations
    * Allow user to define initial particle distribution patterns
    * Allow user to define particles behavior at boundaries
    * Allow user to save simulation parameters presets
    * Add 3D simulation mode