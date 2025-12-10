#pragma once

#include "particlesSoA.cuh"
#include "particlesAoS.cuh"
#include "particle.h"

#include <utility>
#include <cstdint>
#include <cstdlib>

class ComputeBuffers {
public:
    // Buffer sets for previous and next simulation steps, we need two sets to avoid read-write conflicts
    ParticlesSoA prev;
    ParticlesSoA next;
    ParticlesAoS renderBuffer; // buffer for rendering (AoS format)

    // Constructor and destructor
    ComputeBuffers(uint32_t max_particles);
    ~ComputeBuffers();

    // Swap prev and next buffers when a simulation step is complete (better than copying data)
    void swap() {
        std::swap(prev, next);
    }

private:
    // Maximum capacity of particles
    uint32_t capacity;
};
