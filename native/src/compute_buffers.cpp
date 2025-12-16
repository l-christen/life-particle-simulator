#include "compute_buffers.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// This file was made to avoid indirect cuda runtime includes in GDExtension

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

// Constructor that allocates device memory for compute buffers
ComputeBuffers::ComputeBuffers(uint32_t max_particles) {
    capacity = max_particles;

    // Compute SoA buffer sizes
    size_t bufferSoASize = 0;
    bufferSoASize += sizeof(float) * capacity * 4; // x, y, vx, vy
    bufferSoASize += sizeof(uint32_t) * capacity;  // type
    // Allocate prev buffer
    gpuErrchk( cudaMalloc(&d_prev, bufferSoASize) );
    // Allocate next buffer
    gpuErrchk( cudaMalloc(&d_next, bufferSoASize) );
    // Define pointers positions in prev buffer
    size_t offset = 0;
    prev.d_x = (float*)((char*)d_prev + offset);
    offset += sizeof(float) * capacity;
    prev.d_y = (float*)((char*)d_prev + offset);
    offset += sizeof(float) * capacity;
    prev.d_vx = (float*)((char*)d_prev + offset);
    offset += sizeof(float) * capacity;
    prev.d_vy = (float*)((char*)d_prev + offset);
    offset += sizeof(uint32_t) * capacity;
    prev.d_type = (uint32_t*)((char*)d_prev + offset);
    // Define pointers positions in next buffer
    offset = 0;
    next.d_x = (float*)((char*)d_next + offset);
    offset += sizeof(float) * capacity;
    next.d_y = (float*)((char*)d_next + offset);
    offset += sizeof(float) * capacity;
    next.d_vx = (float*)((char*)d_next + offset);
    offset += sizeof(float) * capacity;
    next.d_vy = (float*)((char*)d_next + offset);
    offset += sizeof(uint32_t) * capacity;
    next.d_type = (uint32_t*)((char*)d_next + offset);

    // Allocate device memory for render buffer
    gpuErrchk( cudaMalloc(&renderBuffer.d_particles, sizeof(Particle) * capacity) );

    // Allocate host memory for render buffer
    renderBuffer.h_particles = new Particle[capacity];

    // Initialize number of particles to zero, will be set during simulation initialization
    prev.numParticles = 0;
    next.numParticles = 0;
    renderBuffer.numParticles = 0;
}

// Destructor that frees device memory
ComputeBuffers::~ComputeBuffers() {
    gpuErrchk( cudaFree(d_prev) );

    gpuErrchk( cudaFree(d_next) );

    gpuErrchk( cudaFree(renderBuffer.d_particles) );

    delete[] renderBuffer.h_particles;
}