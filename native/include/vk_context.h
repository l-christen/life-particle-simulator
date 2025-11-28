#pragma once

#include <vulkan/vulkan.h>

namespace godot {
    class RenderingServer;
    class RenderingDevice;
    class VkContext {
        private:
            // Native Vulkan device handle kept private to avoid direct access
            VkPhysicalDevice physical_device = VK_NULL_HANDLE;
            VkDevice device = VK_NULL_HANDLE;
            VkQueue queue = VK_NULL_HANDLE;
            uint32_t queue_family_index = 0;
            VkContext() = default;
        public:
            // Singleton instance
            static VkContext& get() {
                static VkContext instance;
                return instance;
            }
            // Check if Vulkan device is initialized (we don't check for queue family index since it can be 0)
            bool initialized() const { return device != VK_NULL_HANDLE && queue != VK_NULL_HANDLE && physical_device != VK_NULL_HANDLE; }

            // Getter and Setter for VkPhysicalDevice
            VkPhysicalDevice get_physical_device() const { return physical_device; }
            void set_physical_device(VkPhysicalDevice phys_dev) { physical_device = phys_dev; }
            
            // Getter and Setter for VkDevice
            VkDevice get_device() const { return device; }
            void set_device(VkDevice dev) { device = dev; }

            // Getter and Setter for VkQueue
            VkQueue get_queue() const { return queue; }
            void set_queue(VkQueue q) { queue = q; }

            // Getter and Setter for queue family index
            uint32_t get_queue_family_index() const { return queue_family_index; }
            void set_queue_family_index(uint32_t index) { queue_family_index = index; }

            // Vulkan initialization method : This method is called from _process because in _ready the rendering device is not yet ready
            bool init(RenderingServer *rs);
    };
}