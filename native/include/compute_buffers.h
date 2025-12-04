#pragma once

#include "particlesSoA.cuh"

#include <cuda_runtime.h>
#include <cstdint>

class ComputeBuffers {
public:
    // Buffer sets for previous and next simulation steps, we need two sets to avoid read-write conflicts
    ParticlesSoA prev;
    ParticlesSoA next;

    // Constructor that allocates device memory for compute buffers
    ComputeBuffers(uint32_t max_particles) {
        capacity = max_particles;

        // Allocate prev buffers
        cudaMalloc(&prev.d_x, sizeof(float) * capacity);
        cudaMalloc(&prev.d_y, sizeof(float) * capacity);
        cudaMalloc(&prev.d_vx, sizeof(float) * capacity);
        cudaMalloc(&prev.d_vy, sizeof(float) * capacity);
        cudaMalloc(&prev.d_type, sizeof(uint32_t) * capacity);

        // Allocate next buffers
        cudaMalloc(&next.d_x, sizeof(float) * capacity);
        cudaMalloc(&next.d_y, sizeof(float) * capacity);
        cudaMalloc(&next.d_vx, sizeof(float) * capacity);
        cudaMalloc(&next.d_vy, sizeof(float) * capacity);

        next.d_type = prev.d_type; // shared between prev and next

        prev.capacity = capacity;
        next.capacity = capacity;

        // Initialize number of particles to zero, will be set during initialization
        prev.numParticles = 0;
        next.numParticles = 0;
    }

    // Destructor that frees device memory
    ~ComputeBuffers() {
        cudaFree(prev.d_x);
        cudaFree(prev.d_y);
        cudaFree(prev.d_vx);
        cudaFree(prev.d_vy);

        cudaFree(prev.d_type); // shared between prev and next

        cudaFree(next.d_x);
        cudaFree(next.d_y);
        cudaFree(next.d_vx);
        cudaFree(next.d_vy);
    }

    // Swap prev and next buffers when a simulation step is complete (better than copying data)
    void swap() {
        std::swap(prev, next);
    }

private:
    uint32_t capacity;
};
