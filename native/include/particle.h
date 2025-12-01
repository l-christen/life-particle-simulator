#pragma once

#include <cstdint>

struct Particle {
    float x;
    float y;
    uint32_t type;
    uint32_t padding; // we need a padding to align to 16 bytes for cuda
};