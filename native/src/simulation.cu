#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constant memory
// Store particles types in constant memory for faster access in kernels
// It is redundant and read only, so constant memory is a good fit
// Storing default type in json and user may override it with ui
#define NUM_PARTICLE_TYPES 10 // example value, adjust as needed
__constant__ ParticleType d_particleTypes[NUM_PARTICLE_TYPES];

// Initialize simulation
extern "C" void initSimulation() { // placeholder for initialization logic
    // read simulation parameters
    // simulation parameters : number of particles, distribution for each type, 
    // initialize_buffers(); // previous state buffer(SoA), next state buffer(SoA) and render buffer(AoS)
    // set_particle_types(); // copy particle types to constant memory
    // updateParticlePositions<<<numBlocks, blockSize>>>(...); // launch kernel to update particle positions
}

// Kernel to update particle positions based on their types
__global__ void updateParticlePositions(Particle* particles, int numParticles, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        // Simulation logic based on particle type
    }
}

/*
utils :
void swap_buffers_handle(previous_buffer, next_buffer)
auto& get_neighbors(particle_index, buffer)
void compute_new_velocity(particle, neighbors)
void update_position(particle, deltaTime)
void initialize_buffers(parameters)
void set_particle_types(file_path_json)
*/