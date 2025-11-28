#pragma once

#include <cstdint>

struct Particle {
    float x;
    float y;
    uint32_t type;
    uint32_t _pad; // Padding for alignment for Vulkan buffers
};