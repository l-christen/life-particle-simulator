#include "vk_context.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <vulkan/vulkan.h>

using namespace godot;

// Vulkan initialization method : This method is called from _process because in _ready the rendering device is not yet ready
bool VkContext::init(RenderingServer *rs) {
    if (!rs) {
        UtilityFunctions::print("RenderingServer not available.");
        return false;
    }

    // Get RenderingDevice from RenderingServer singleton
    RenderingDevice *rd = rs->get_rendering_device();
    if (!rd) {
        UtilityFunctions::print("RenderingDevice not available.");
        return false;
    }

    // Retrieve Vulkan logical device
    uint64_t raw_device = rd->get_driver_resource(
        RenderingDevice::DRIVER_RESOURCE_LOGICAL_DEVICE,
        RID(),
        0
    );
    if (raw_device == 0) {
        UtilityFunctions::print("Failed to fetch Vulkan device handle!");
        return false;
    }

    // Store Vulkan device handle
    this->set_device(reinterpret_cast<VkDevice>(raw_device));
    // Print Vulkan device handle, we need to cast it to uint64_t for printing
    UtilityFunctions::print("Vulkan device OK: ", (uint64_t)this->get_device());
    return true;
}