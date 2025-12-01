#pragma once

enum ParticleType {
    PARTICLE_TYPE_A,
    PARTICLE_TYPE_B,
    PARTICLE_TYPE_C
};

struct Particle {
    float x;
    float y;
    ParticleType type;
    int padding; // we need a padding to align to 16 bytes for cuda
};