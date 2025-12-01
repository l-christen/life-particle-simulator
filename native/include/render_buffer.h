#pragma once

#include "particlesAoS.cuh"

#include <cuda_runtime.h>

class RenderBuffer {
    public:
        // Constructor that allocates pinned host memory for particles
        RenderBuffer(size_t max_particles) {
            capacity = max_particles;

            // Allocate pinned host memory for particles
            cudaHostAlloc(&h_particles, sizeof(Particle) * capacity, cudaHostAllocMapped);

            // Get device pointers for the mapped host memory
            cudaHostGetDevicePointer(&d_particles, h_particles, 0);
            
            // Initialize ParticlesAoS structure with device and host pointers
            particlesAoS.d_particles = d_particles;
            particlesAoS.h_particles = h_particles;

            particlesAoS.capacity = capacity;
        }

        // Destructor that frees pinned host memory
        ~RenderBuffer() {
            cudaFreeHost(h_particles);
        }

        private:
        Particle* h_particles; // host pointer to pinned memory
        Particle* d_particles; // device pointer to mapped host memory

        ParticlesAoS particlesAoS; // structure holding pointers and capacity
};