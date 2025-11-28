#include "cuda_particles_renderer.h"
#include "vk_context.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

using namespace godot;

// Custom methods have to be binded here in order to be accessible from GDScript
void CudaParticlesRenderer::_bind_methods() {
}

// Constructor and Destructor
CudaParticlesRenderer::CudaParticlesRenderer() {
}

CudaParticlesRenderer::~CudaParticlesRenderer() {
}

void CudaParticlesRenderer::_ready() {
    UtilityFunctions::print("CudaParticlesRenderer is ready!");
}

void CudaParticlesRenderer::_process(double delta) {
    RenderingServer *rs = RenderingServer::get_singleton();
    // Initialize Vulkan context if not already done
    if (!VkContext::get().initialized()) {
        if (!VkContext::get().init(rs)) {
            UtilityFunctions::print("Failed to initialize Vulkan context.");
            return;
        }
    }
}