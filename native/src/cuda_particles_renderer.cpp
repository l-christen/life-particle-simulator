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

    // Initialize particle buffer once
    if (!buffer_initialized) {
        // Let's try to create a 1000 particles with CPU data
        const size_t particle_count = 1000;
        // Resize CPU-side particle vector
        particles_cpu.resize(particle_count);

        // Create some dummy particle data
        for (size_t i = 0; i < particle_count; ++i) {
            particles_cpu[i].x = float(i % 100);
            particles_cpu[i].y = float(i / 100);
            particles_cpu[i].type = i % 4;
            particles_cpu[i]._pad = 0;
        }

        // Create native particle buffer and upload data
        native_buffer = std::make_unique<NativeParticleBuffer>(particle_count);
        native_buffer->upload(particles_cpu.data(), particles_cpu.size());

        UtilityFunctions::print("Native particle buffer created and uploaded.");
        buffer_initialized = true;
    }
}