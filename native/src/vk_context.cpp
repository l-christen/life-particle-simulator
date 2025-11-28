#include "vk_context.h"

#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/rendering_device.hpp>

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

    // Get Vulkan physical device from RenderingDevice
    uint64_t raw_physical_device = rd->get_driver_resource(
        RenderingDevice::DRIVER_RESOURCE_VULKAN_PHYSICAL_DEVICE,
        RID(),
        0
    );
    if (raw_physical_device == 0) {
        UtilityFunctions::print("Failed to fetch Vulkan physical device handle!");
        return false;
    }

    // Get Vulkan queue from RenderingDevice
    uint64_t raw_queue = rd->get_driver_resource(
        RenderingDevice::DRIVER_RESOURCE_VULKAN_QUEUE,
        RID(),
        0
    );
    if (raw_queue == 0) {
        UtilityFunctions::print("Failed to fetch Vulkan queue handle!");
        return false;
    }

    // Get queue family index from RenderingDevice
    uint64_t queue_family_index = rd->get_driver_resource(
        RenderingDevice::DRIVER_RESOURCE_VULKAN_QUEUE_FAMILY_INDEX,
        RID(),
        0
    );

    // Store Vulkan physical device handle
    this->set_physical_device(reinterpret_cast<VkPhysicalDevice>(raw_physical_device));
    // Store Vulkan device handle
    this->set_device(reinterpret_cast<VkDevice>(raw_device));
    // Store Vulkan queue handle
    this->set_queue(reinterpret_cast<VkQueue>(raw_queue));
    // Store queue family index
    this->set_queue_family_index((uint32_t)queue_family_index);
    // Print Vulkan device handle, we need to cast it to uint64_t for printing
    UtilityFunctions::print("Vulkan physical device OK: ", (uint64_t)this->get_physical_device());
    UtilityFunctions::print("Vulkan device OK: ", (uint64_t)this->get_device());
    UtilityFunctions::print("Vulkan queue OK: ", (uint64_t)this->get_queue());
    UtilityFunctions::print("Vulkan queue family index OK: ", this->get_queue_family_index());
    return this->initialized();
}