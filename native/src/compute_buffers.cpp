#include "compute_buffers.h"

#include <cuda_runtime.h>

// This file was made to avoid indirect cuda runtime includes in GDExtension

// Constructor that allocates device memory for compute buffers
ComputeBuffers::ComputeBuffers(uint32_t max_particles) {
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

    // Allocate device memory for render buffer
    cudaMalloc(&renderBuffer.d_particles, sizeof(Particle) * capacity);

    // Allocate host memory for render buffer
    renderBuffer.h_particles = new Particle[capacity];

    next.d_type = prev.d_type; // shared between prev and next

    prev.capacity = capacity;
    next.capacity = capacity;
    renderBuffer.capacity = capacity;

    // Initialize number of particles to zero, will be set during simulation initialization
    prev.numParticles = 0;
    next.numParticles = 0;
    renderBuffer.numParticles = 0;
}

// Destructor that frees device memory
ComputeBuffers::~ComputeBuffers() {
    cudaFree(prev.d_x);
    cudaFree(prev.d_y);
    cudaFree(prev.d_vx);
    cudaFree(prev.d_vy);

    cudaFree(prev.d_type); // shared between prev and next

    cudaFree(next.d_x);
    cudaFree(next.d_y);
    cudaFree(next.d_vx);
    cudaFree(next.d_vy);

    cudaFree(renderBuffer.d_particles);

    delete[] renderBuffer.h_particles;
}