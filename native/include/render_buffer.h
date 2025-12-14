#pragma once

#include "particlesAoS.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <utility>

/*
For now this class is not used because mapped memory might not allow to compile the extension.
*/

class RenderBuffer {
    public:
        // Constructor that allocates pinned host memory for particles
        RenderBuffer(uint32_t max_particles) {

            // Allocate pinned host memory for particles and mapped flags
            cudaHostAlloc(&h_particles, sizeof(Particle) * max_particles, cudaHostAllocMapped);

            // Get device pointers for the mapped host memory
            cudaHostGetDevicePointer(&d_particles, h_particles, 0);
            
            // Initialize ParticlesAoS structure with device and host pointers
            particlesAoS.d_particles = d_particles;
            particlesAoS.h_particles = h_particles;
            
            // Set capacity and initialize number of particles to zero
            particlesAoS.capacity = max_particles;
            particlesAoS.numParticles = 0;
        }

        // Destructor that frees pinned host memory
        ~RenderBuffer() {
            cudaFreeHost(h_particles);
        }

        ParticlesAoS particlesAoS; // structure particles buffer

        private:
        Particle* h_particles; // host pointer to pinned memory
        Particle* d_particles; // device pointer to mapped host memory
};