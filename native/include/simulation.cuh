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
    uint32_t numRed,
    uint32_t numBlue,
    uint32_t numGreen,
    uint32_t numYellow,
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

void setSimulationRules(
    const float* rules
);

void setSimulationRadiusOfInfluence(
    const float* radiusOfInfluence
);

#ifdef __cplusplus
}
#endif
