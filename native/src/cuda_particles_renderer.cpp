#include "cuda_particles_renderer.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <vulkan/vulkan.h>

using namespace godot;

// Custom methods have to be binded here in order to be accessible from GDScript
void CudaParticlesRenderer::_bind_methods() {
    // Bind _init_vulkan method
    ClassDB::bind_method(D_METHOD("_init_vulkan"), &CudaParticlesRenderer::_init_vulkan);
}

// Constructor and Destructor
CudaParticlesRenderer::CudaParticlesRenderer() {
}

CudaParticlesRenderer::~CudaParticlesRenderer() {
}

// Vulkan initialization method : This method is called from _process because in _ready the rendering device is not yet ready
void CudaParticlesRenderer::_init_vulkan() {
    UtilityFunctions::print("Initializing Vulkan...");

    // Get RenderingServer singleton
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        UtilityFunctions::print("RenderingServer not available.");
        return;
    }

    // Get RenderingDevice from RenderingServer singleton
    RenderingDevice *rd = rs->get_rendering_device();
    if (!rd) {
        UtilityFunctions::print("RenderingDevice not available.");
        return;
    }

    // Retrieve Vulkan logical device
    uint64_t raw_device = rd->get_driver_resource(
        RenderingDevice::DRIVER_RESOURCE_LOGICAL_DEVICE,
        RID(),
        0
    );
    if (raw_device == 0) {
        UtilityFunctions::print("Failed to fetch Vulkan device handle!");
        return;
    }

    // Store Vulkan instance handle
    vk_instance = reinterpret_cast<VkInstance>(raw_device);
    // Print Vulkan instance handle, we need to cast it to uint64_t for printing
    UtilityFunctions::print("Vulkan instance OK: ", (uint64_t)vk_instance);
}

void CudaParticlesRenderer::_ready() {
    UtilityFunctions::print("CudaParticlesRenderer is ready!");
}

void CudaParticlesRenderer::_process(double delta) {
    // Initialize Vulkan if not already done
    if (!vk_initialized) {
        _init_vulkan();
        vk_initialized = true;
    }

}