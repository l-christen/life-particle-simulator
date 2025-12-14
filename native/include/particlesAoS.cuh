#pragma once

#include "particle.h"
#include <cstdint>

/*
This structure represents a renderer buffer using an Array of Structures (AoS) layout for particles.
The choice of AoS Layout was made to optimize memory access patterns for rendering operations from the CPU side.
*/

struct ParticlesAoS {
    uint32_t numParticles; // number of particles in the buffer

    Particle* d_particles; // device pointer to particle data
    Particle* h_particles; // host pointer to particle data
};