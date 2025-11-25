# 01 - Simulation app
This file describes the considerations that led to the choice of the project's overall architecture.
## Context
The goal of the project is to create an application that simulates living particles. Calculations must be performed in real time at a frequency of 60Hz, and simulation parameters must be accessible and modifiable by the user.
## Options considered
- Webapp
- Godot + Vulkan
- Godot + Cuda
## Webapp
The advantage of a web app is portability. It would be accessible anywhere without installation. There are libraries such as WebGPU that allow interaction with graphics hardware for simulation calculations, but it remains a relatively high-level API that would probably not allow heavy simulations to be run[[1]](https://medium.com/source-true/webgpu-performance-is-it-what-we-expect-b1c96b1705e1).
## Godot + Vulkan
With a C++ extension for Godot[[2]](https://docs.godotengine.org/en/4.4/tutorials/scripting/gdextension/gdextension_cpp_example.html), it seems possible to retrieve the VulkanDevice from the engine[[3]](https://docs.godotengine.org/en/stable/classes/class_renderingdevice.html#class-renderingdevice-method-get-driver-resource) and create a shared buffer with Godot, perform simulation calculations with native Vulkan, and display the results with a Godot shader without memory copying between the CPU and GPU. The advantage of this architecture is that Vulkan code can be executed on any GPU with performance close to that obtained with proprietary code such as Cuda.
## Godot + Cuda
The last architecture considered would be to follow the same steps as the previous one but taking advantage of Cuda Vulkan interoperability[[4]](https://www.gpultra.com/blog/vulkan-cuda-memory-interoperability/) by performing the calculations with Cuda. The main disadvantage of this method is the requirement to use an Nvidia GPU. The two main advantages of this method are the optimizations offered by nvcc and the fact that I have already been introduced to Cuda.