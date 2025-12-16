// This simulation kernel is a first simple implementation of a particle system
// inspired by https://www.youtube.com/watch?v=0Kx4Y9TVMGg

#include "particlesAoS.cuh"
#include "particlesSoA.cuh"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>

// Macro to check for CUDA errors found there : https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// Constant memory
// Store particles rules in constant memory for faster access in kernels
// It is redundant and read only, so constant memory is a good fit
#define NUM_PARTICLE_TYPES 4 // static value for now

// Interaction rules between particle types (matrix flattened)
__constant__ float particleRules[NUM_PARTICLE_TYPES * NUM_PARTICLE_TYPES];
// Store radius of influence for each particle type (squared values to avoid redundant sqrt calculations)
__constant__ float radiusOfInfluence[NUM_PARTICLE_TYPES];

// Define a Vec2 structure
struct Vec2 {
    float x;
    float y;
};

// This function initializes the simulation with random particle positions and zero velocities
// It can be improved by generating the randomness on the GPU directly
extern "C" void initSimulation(
    ParticlesSoA prev, // previous state
    ParticlesSoA next, // next state
    ParticlesAoS* render, // render buffer
    uint32_t numRed, // number of red particles
    uint32_t numBlue, // number of blue particles
    uint32_t numGreen, // number of green particles
    uint32_t numYellow, // number of yellow particles
    float width, // simulation width
    float height // simulation height
) {
    // Total number of particles
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
        // Affect random position and zero velocity
        h_x[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / width));
        h_y[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / height));
        h_vx[i] = 0.0f;
        h_vy[i] = 0.0f;
        // Define type based on required quantity
        uint32_t type = 0;
        if (countRed < numRed) {
            type = 0; // Red
            countRed++;
        }
        else if (countBlue < numBlue) {
            type = 1; // Blue
            countBlue++;
        }
        else if (countGreen < numGreen) {
            type = 2; // Green
            countGreen++;
        }
        else if (countYellow < numYellow) {
            type = 3; // Yellow
            countYellow++;
        }
        h_type[i] = type;
        
        h_particles[i] = { h_x[i], h_y[i], h_type[i], 0 };
    }

    // Copy to device
    gpuErrchk( cudaMemcpy(prev.d_x, h_x, sizeof(float) * numParticles, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(prev.d_y, h_y, sizeof(float) * numParticles, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(prev.d_vx, h_vx, sizeof(float) * numParticles, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(prev.d_vy, h_vy, sizeof(float) * numParticles, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(prev.d_type, h_type, sizeof(uint32_t) * numParticles, cudaMemcpyHostToDevice) );
    
    gpuErrchk( cudaMemcpy(next.d_x, h_x, sizeof(float) * numParticles, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(next.d_y, h_y, sizeof(float) * numParticles, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(next.d_vx, h_vx, sizeof(float) * numParticles, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(next.d_vy, h_vy, sizeof(float) * numParticles, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(next.d_type, h_type, sizeof(uint32_t) * numParticles, cudaMemcpyHostToDevice) );
    
    gpuErrchk( cudaMemcpy(render->d_particles, h_particles, sizeof(Particle) * numParticles, cudaMemcpyHostToDevice) );


    // Clean up
    delete[] h_x;
    delete[] h_y;
    delete[] h_vx;
    delete[] h_vy;
    delete[] h_type;
    delete[] h_particles;
    
    // Set number of particles
    prev.numParticles = numParticles;
    next.numParticles = numParticles;
    render->numParticles = numParticles;
}

// Get the distance squared between two particles
__device__ float distance_squared(float dx, float dy) {
    float distanceSquared = dx * dx + dy * dy;
    return distanceSquared;
}

// Get the normalized vector between two particles, allow us to decompose forces without trigonometric functions
__device__ Vec2 normalized_vector_between_particles(float dx, float dy) {
    float inv_length = rsqrtf(dx * dx + dy * dy); // reciprocal of the length, using rsqrtf for performance
    return { dx * inv_length, dy * inv_length };
}

// Kernel to update particle positions based on their types
__global__ void updateParticlePositions(
    ParticlesSoA prev, // previous state
    ParticlesSoA next, // next state
    Particle* render_d_particles, // render buffer on device
    int numParticles, // number of particles
    float width, // simulation width
    float height, // simulation height
    float viscosity, // viscosity factor
    float deltaTime // time step
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        float ax = 0.0f; // accumulated acceleration for x
        float ay = 0.0f; // accumulated acceleration for y
        for (int j = 0; j < numParticles; j++) { // loop over all particles to compute interactions
            if (idx != j) {
                float dx = prev.d_x[j] - prev.d_x[idx];
                float dy = prev.d_y[j] - prev.d_y[idx];
                // Apply periodic boundary conditions (toroidal space)
                if (dx > width / 2) dx -= width;
                if (dx < -width / 2) dx += width;
                if (dy > height / 2) dy -= height;
                if (dy < -height / 2) dy += height;

                float dist_sq = distance_squared(dx, dy); // squared distance
                // Check if distance squared is smaller than radius of influence squared
                if (dist_sq < radiusOfInfluence[prev.d_type[idx]]) {
                    Vec2 dir = normalized_vector_between_particles(dx, dy); // normalized direction vector to decompose force
                    // Apply force based on particle types and distance with an adaptation of Newton's law of universal gravitation
                    float force = particleRules[prev.d_type[idx] * NUM_PARTICLE_TYPES + prev.d_type[j]] / (dist_sq + 225.0f); // add a softening to avoid extreme forces at small distances
                    // Clamp force to avoid extreme values
                    force = fminf(fmaxf(force, -100.0f), 100.0f);
                    ax += force * dir.x; // decompose force into x and accumulate
                    ay += force * dir.y; // decompose force into y and accumulate
                }
            }
        }
        // Update velocity (v = v0 + a * t)
        next.d_vx[idx] = prev.d_vx[idx] + ax * deltaTime;
        next.d_vy[idx] = prev.d_vy[idx] + ay * deltaTime;

        // Apply damping (viscosity) might be overkill since we already clamp the forces
        float damping = expf(-viscosity * deltaTime); // damping increase with viscosity and time
        next.d_vx[idx] *= damping;
        next.d_vy[idx] *= damping;

        // Max speed 600.0f
        // Compute speed scalar squared
        float speed_sq = next.d_vx[idx] * next.d_vx[idx] + next.d_vy[idx] * next.d_vy[idx] + 1e-6f;
        if (speed_sq > 600.0f * 600.0f) {
            float inv_speed = rsqrtf(speed_sq); // reciprocal of the speed
            // Normalize and scale to max speed
            next.d_vx[idx] = next.d_vx[idx] * inv_speed * 600.0f;
            next.d_vy[idx] = next.d_vy[idx] * inv_speed * 600.0f;
        }

        // Update position with boundary checks (toroidal space)
        float temp_x = prev.d_x[idx] + next.d_vx[idx] * deltaTime;
        float temp_y = prev.d_y[idx] + next.d_vy[idx] * deltaTime;
        while (temp_x < 0) temp_x += width;
        while (temp_y < 0) temp_y += height;
        next.d_x[idx] = fmodf(temp_x, width);
        next.d_y[idx] = fmodf(temp_y, height);
        // Update render buffer
        render_d_particles[idx] = {
            next.d_x[idx],
            next.d_y[idx],
            prev.d_type[idx],
            0 // padding
        };
    }
}

// Wrapper function to lauch the kernel from Godot Extension
extern "C" void runSimulationStep(
    ParticlesSoA prev, // previous state
    ParticlesSoA next, // next state
    ParticlesAoS* render, // render buffer
    int numParticles, // number of particles
    float width, // simulation width
    float height, // simulation height
    float viscosity, // viscosity factor
    float deltaTime // time step
) {
    // Random block and grid size for now, need to be optimized later
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    updateParticlePositions<<<blocksPerGrid, threadsPerBlock>>>(
        prev, // previous state
        next, // next state
        render->d_particles, // render buffer on device
        numParticles, // number of particles
        width, // simulation width
        height, // simulation height
        viscosity, // viscosity factor
        deltaTime // time step
    );

    // Check for kernel launch errors
    gpuErrchk( cudaPeekAtLastError() );

    // Synchronize to ensure kernel completion
    gpuErrchk( cudaDeviceSynchronize() );

    // Copy render buffer back to host
    gpuErrchk( cudaMemcpy(render->h_particles, render->d_particles, sizeof(Particle) * numParticles, cudaMemcpyDeviceToHost) );
}

// Function to set simulation rules from host
extern "C" void setSimulationRules(const float* rules) {
    gpuErrchk( cudaMemcpyToSymbol(particleRules, rules, sizeof(float) * NUM_PARTICLE_TYPES * NUM_PARTICLE_TYPES) );
}

// Function to set radius of influence from host
extern "C" void setSimulationRadiusOfInfluence(const float* radius) {
    // Square the radius values before sending to device
    float* sq_radius = new float[NUM_PARTICLE_TYPES];
    for (int i = 0; i < NUM_PARTICLE_TYPES; i++) {
        sq_radius[i] = radius[i] * radius[i];
    }

    // Store squared radius in constant memory to avoid computing square roots in the kernel
    gpuErrchk( cudaMemcpyToSymbol(radiusOfInfluence, sq_radius, sizeof(float) * NUM_PARTICLE_TYPES) );

    // Clean up
    delete[] sq_radius;
}