#include "cuda_particles_renderer.h"
#include <godot_cpp/core/class_db.hpp>

using namespace godot;

void CudaParticlesRenderer::_bind_methods() {
}

CudaParticlesRenderer::CudaParticlesRenderer() {
}

CudaParticlesRenderer::~CudaParticlesRenderer() {
}

void CudaParticlesRenderer::_ready() {
    UtilityFunctions::print("CudaParticlesRenderer is ready!");
}

void CudaParticlesRenderer::_process(double delta) {
    UtilityFunctions::print("Process tick: ", delta);
}