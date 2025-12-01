#pragma once

#include "particle.h"

#include <cuda_runtime.h>
#include <cuda/std/span>
#include <cstdint>

/*
This structure represents a renderer buffer using an Array of Structures (AoS) layout for particles.
The choice of AoS Layout was made to optimize memory access patterns for rendering operations from the CPU side.
*/

struct ParticlesAoS {
    uint32_t numParticles; // number of particles in the buffer
    uint32_t capacity;     // maximum number of particles the buffer can hold

    Particle* d_particles; // device pointer to particle data
    Particle* h_particles; // host pointer to particle data

    // Accessor for particle at index i
    __host__ __device__ Particle& particle_i(uint32_t i) const {
        #ifdef __CUDA_ARCH__
            return d_particles[i];
        #else
            return h_particles[i];
        #endif
    }
};