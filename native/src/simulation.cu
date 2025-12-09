
#include "particlesAoS.cuh"
#include "particlesSoA.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>


// This simulation kernel is a first simple implementation of a particle system
// inspired by https://www.youtube.com/watch?v=0Kx4Y9TVMGg


// Constant memory
// Store particles rules in constant memory for faster access in kernels
// It is redundant and read only, so constant memory is a good fit
#define NUM_PARTICLE_TYPES 4 // static value for now
__constant__ float particleRules[NUM_PARTICLE_TYPES * NUM_PARTICLE_TYPES];
// Store radius of influence for each particle type
__constant__ float radiusOfInfluence[NUM_PARTICLE_TYPES];

// Define a Vec2 structure
struct Vec2 {
    float x;
    float y;
};

// This function has to be redone properly, this is for testing purposes only
extern "C" void initSimulation(
    ParticlesSoA* prev,
    ParticlesSoA* next,
    ParticlesAoS* render,
    uint32_t numRed,
    uint32_t numBlue,
    uint32_t numGreen,
    uint32_t numYellow,
    float width,
    float height
) {
    uint32_t numParticles = numRed + numBlue + numGreen + numYellow;

    // Allocate host arrays
    float* h_x = new float[numParticles];
    float* h_y = new float[numParticles];
    float* h_vx = new float[numParticles];
    float* h_vy = new float[numParticles];
    uint32_t* h_type = new uint32_t[numParticles];
    Particle* h_particles = new Particle[numParticles];

    // Counts for each type
    uint32_t countRed = 0;
    uint32_t countBlue = 0;
    uint32_t countGreen = 0;
    uint32_t countYellow = 0;

    // Initialize on host
    for (int i = 0; i < numParticles; i++) {
        h_x[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / width));
        h_y[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / height));
        h_vx[i] = 0.0f;
        h_vy[i] = 0.0f;
        uint32_t type = 0;
        if (countRed < numRed) {
            type = 1; // Red
            countRed++;
        }
        else if (countBlue < numBlue) {
            type = 2; // Blue
            countBlue++;
        }
        else if (countGreen < numGreen) {
            type = 3; // Green
            countGreen++;
        }
        else if (countYellow < numYellow) {
            type = 4; // Yellow
            countYellow++;
        }
        h_type[i] = type;
        
        h_particles[i] = { h_x[i], h_y[i], h_type[i], 0 };
    }

    // Copy to device
    cudaMemcpy(prev->d_x, h_x, sizeof(float) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(prev->d_y, h_y, sizeof(float) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(prev->d_vx, h_vx, sizeof(float) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(prev->d_vy, h_vy, sizeof(float) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(prev->d_type, h_type, sizeof(uint32_t) * numParticles, cudaMemcpyHostToDevice);
    
    cudaMemcpy(next->d_x, h_x, sizeof(float) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(next->d_y, h_y, sizeof(float) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(next->d_vx, h_vx, sizeof(float) * numParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(next->d_vy, h_vy, sizeof(float) * numParticles, cudaMemcpyHostToDevice);
    
    cudaMemcpy(render->d_particles, h_particles, sizeof(Particle) * numParticles, cudaMemcpyHostToDevice);


    // Clean up
    delete[] h_x;
    delete[] h_y;
    delete[] h_vx;
    delete[] h_vy;
    delete[] h_type;
    delete[] h_particles;
    
    prev->numParticles = numParticles;
    next->numParticles = numParticles;
    render->numParticles = numParticles;
}

// Get the distance squared between two particles
__device__ float distance_squared(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float distanceSquared = dx * dx + dy * dy;
    return distanceSquared;
}

// Get the normalized vector between two particles, allow us to decompose forces without trigonometric functions
__device__ Vec2 normalized_vector_between_particles(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float inv_length = rsqrtf(dx * dx + dy * dy);
    return { dx * inv_length, dy * inv_length };
}

// Kernel to update particle positions based on their types
__global__ void updateParticlePositions(
    ParticlesSoA prev,
    ParticlesSoA next,
    ParticlesAoS render_buffer,
    int numParticles,
    float width,
    float height,
    float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        float ax = 0.0f; // accumulated acceleration for x
        float ay = 0.0f; // accumulated acceleration for y
        for (int j = 0; j < numParticles; j++) { // loop over all particles to compute interactions
            if (idx != j) {
                float dist_sq = distance_squared(prev.d_x[idx], prev.d_y[idx], prev.d_x[j], prev.d_y[j]);
                if (dist_sq < (radiusOfInfluence[prev.d_type[j]] * radiusOfInfluence[prev.d_type[j]])) {
                    Vec2 dir = normalized_vector_between_particles(prev.d_x[idx], prev.d_y[idx], prev.d_x[j], prev.d_y[j]);
                    // Apply force based on particle types and distance with an adaptation of Newton's law of universal gravitation
                    float force = particleRules[prev.d_type[idx] * NUM_PARTICLE_TYPES + prev.d_type[j]] / (dist_sq + 0.0001f); // avoid division by zero
                    ax += force * dir.x; // decompose force into x and accumulate
                    ay += force * dir.y; // decompose force into y and accumulate
                }
            }
        }
        // Update velocity (v = v0 + a * t)
        next.d_vx[idx] = prev.d_vx[idx] + ax * deltaTime;
        next.d_vy[idx] = prev.d_vy[idx] + ay * deltaTime;
        // Update position with boundary checks (toroidal space)
        next.d_x[idx] = fmodf(prev.d_x[idx] + next.d_vx[idx] * deltaTime + width, width);
        next.d_y[idx] = fmodf(prev.d_y[idx] + next.d_vy[idx] * deltaTime + height, height);
        // Update render buffer
        render_buffer.d_particles[idx] = {
            next.d_x[idx],
            next.d_y[idx],
            prev.d_type[idx],
            0 // padding
        };
    }
}

// Wrapper function to lauch the kernel from Godot Extension
extern "C" void runSimulationStep(
    ParticlesSoA* prev,
    ParticlesSoA* next,
    ParticlesAoS* render,
    int numParticles,
    float width,
    float height,
    float deltaTime
) {
    // "Random" block and grid size for now, need to be optimized later
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel in the created stream
    updateParticlePositions<<<blocksPerGrid, threadsPerBlock>>>(
        *prev,
        *next,
        *render,
        numParticles,
        width,
        height,
        deltaTime
    );


    cudaDeviceSynchronize();

    cudaMemcpy(render->h_particles, render->d_particles, sizeof(Particle) * numParticles, cudaMemcpyDeviceToHost);
}

extern "C" void setSimulationRules(const float* rules) {
    cudaMemcpyToSymbol(particleRules, rules, sizeof(float) * NUM_PARTICLE_TYPES * NUM_PARTICLE_TYPES);
}

extern "C" void setSimulationRadiusOfInfluence(const float* radiusOfInfluence) {
    cudaMemcpyToSymbol(radiusOfInfluence, radiusOfInfluence, sizeof(float) * NUM_PARTICLE_TYPES);
}