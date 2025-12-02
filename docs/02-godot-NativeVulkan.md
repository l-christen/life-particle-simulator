# Why direct use of native Vulkan with Godot is unstable
This document details why my initial architectural choice led me to a dead end.
## Introduction
In my architecture choice document, I explained that I wanted to use Godot with native Vulkan to perform simulation calculations on the GPU by taking advantage of Cuda Vulkan interoperability. The idea was to create a shared buffer between Godot and native Vulkan with Cuda interop to avoid memory copies between the CPU and GPU. However, after several attempts, I realized that this approach is unstable and can lead to crashes or undefined behavior.

It is certainly possible to create these interactions between Godot and native Vulkan, but I set myself a time limit (3 days) and decided not to fork Godot to modify the engine so as not to get into concepts that I don't master.

It should also be noted that I have a basic understanding of how a GPU works, acquired through some Cuda programming. I had never worked with Vulkan before. The Vulkan documentation is not easy to get to grips with, and the Godot documentation does not help with low-level details.
## Context
With Godot's public API, I was able to obtain the VkPhysicalDevice, VkLogicalDevice, VkQueue, and its QueueFamilyIndex.

So far, all the elements necessary for copying a native Vulkan buffer seem to be present, but there are some subtleties.

## Public API limitations
- The Godot public API allows you to create a buffer needed to hold data from the native buffer. However, we have no control over the flags assigned to the buffer, and without the necessary flags, it is impossible to copy data into it.
- The Godot public API allows you to retrieve a CommandQueue from its VulkanDevice and also its QueueFamilyIndex. Unfortunately, the only queue we can retrieve is Godot's main rendering queue, which is type 0 (graphics + compute). Even though the graphics queue technically supports transfer operations, using it from a GDExtension to submit our own command buffers interferes with Godot’s internal render pipeline. Since we have no control over synchronization, this leads to undefined behaviour and crashes. And it is not possible to create your own queue because this operation is performed during the initialization of the VkDevice, over which we have no control without forking Godot.

## Backup plan
Without changing the frontend, the only solution that seems consistent with my knowledge and the time I have available is:
- Perform the simulation on the GPU with Cuda
- Copy the data to the CPU
- Transfer it from the CPU to Godot's MultiMeshInstance2d

I discovered three mechanisms that allow me to streamline the data copy pipeline.
- Pinned Memory: This allows you to pin the host memory that will contain the particle buffer, which means that the OS cannot modify the pointer to the buffer.
- Mapped Memory: Mapped pinned memory provides a device pointer that refers to the same physical host memory. The GPU can write it directly over PCIe, without requiring an explicit cudaMemcpy
- WriteCombined: Because my simulation writes particle states sequentially and the CPU reads the buffer only once per frame, WriteCombined memory is a perfect fit. It avoids cache pollution, improves write throughput, and makes GPU-CPU streaming more efficient without any downside in this access pattern.

The combination of these three mechanisms allows the CUDA kernel write particle updates directly into mapped host memory, avoiding the explicit GPU-CPU copy step, which is usually the bottleneck.

This solution is not optimal because the simulations steps have to go through the CPU memory. However, it avoids extra memcpy steps and I hope it remains fast enough for my needs. It is probably a viable solution that allows me to continue the project without getting bogged down in low-level Vulkan details that I do not master.

