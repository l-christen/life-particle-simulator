#pragma once

#include "particlesSoA.cuh"
#include "particlesAoS.cuh"
#include "particle.h"
#include <cstdint>
#include <cstdlib>
#include <utility>

class ComputeBuffers {
public:
    // Buffers (SoA) for previous and next simulation steps, we need two buffers to avoid read-write conflicts
    ParticlesSoA prev;
    ParticlesSoA next;
    ParticlesAoS renderBuffer; // buffer for rendering (AoS format)

    // Constructor and destructor
    ComputeBuffers(uint32_t max_particles);
    ~ComputeBuffers();

    // Swap prev and next buffers when a simulation step is complete (better than copying data)
    void swap() {
        // Swap the buffer pointers
        std::swap(prev.d_x, next.d_x);
        std::swap(prev.d_y, next.d_y);
        std::swap(prev.d_vx, next.d_vx);
        std::swap(prev.d_vy, next.d_vy);
        std::swap(prev.d_type, next.d_type);
        std::swap(prev.numParticles, next.numParticles);
    }

private:
    // Maximum capacity of particles
    uint32_t capacity;

    // GPU SoA buffer structures
    void* d_prev = nullptr;
    void* d_next = nullptr;
};