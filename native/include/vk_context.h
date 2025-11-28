#pragma once

#include <vulkan/vulkan.h>

namespace godot {
    class RenderingServer;
    class RenderingDevice;
    class VkContext {
        private:
            // Native Vulkan device handle kept private to avoid direct access
            VkDevice device = VK_NULL_HANDLE;
            VkContext() = default;
        public:
            // Singleton instance
            static VkContext& get() {
                static VkContext instance;
                return instance;
            }
            // Check if Vulkan device is initialized
            bool initialized() const { return device != VK_NULL_HANDLE; }

            // Getter and Setter for VkDevice
            VkDevice get_device() const { return device; }
            void set_device(VkDevice dev) { device = dev; }

            // Vulkan initialization method : This method is called from _process because in _ready the rendering device is not yet ready
            bool init(RenderingServer *rs);
    };
}