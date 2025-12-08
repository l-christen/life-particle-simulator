#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/*
This structure represents a compute buffer using a Structure of Arrays (SoA) layout for particles.
The choice of SoA Layout was made to optimize memory access patterns for kernel operations on the GPU side.
For example, when we want to get the neighbors of a particle, we only need to access the position arrays (x and y),
which are stored contiguously in memory, leading to better coalesced memory accesses.
*/

struct ParticlesSoA {
    uint32_t numParticles; // number of particles in the buffer
    uint32_t capacity;     // maximum number of particles the buffer can hold

    float* d_x;           // device pointer to x positions
    float* d_y;           // device pointer to y positions
    float* d_vx;          // device pointer to x velocities
    float* d_vy;          // device pointer to y velocities
    uint32_t* d_type;     // device pointer to particle types
};