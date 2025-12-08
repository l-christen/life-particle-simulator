#pragma once

#include "particlesSoA.cuh"
#include "particlesAoS.cuh"

// This file declares the interface for the simulation functions
// inspired by P.A. Mudry ray tracer
#ifdef __cplusplus
extern "C" {
#endif

void initSimulation(
    ParticlesSoA* prev,
    ParticlesSoA* next,
    ParticlesAoS* render,
    const float* rules,
    const float* radiusOfInfluence,
    int numParticles,
    int numTypes,
    float width,
    float height
);

void runSimulationStep(
    ParticlesSoA* prev,
    ParticlesSoA* next,
    ParticlesAoS* render,
    int numParticles,
    float width,
    float height,
    float deltaTime
);

#ifdef __cplusplus
}
#endif
