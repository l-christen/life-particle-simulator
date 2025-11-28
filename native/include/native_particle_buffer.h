#include "particle.h"
#include "vk_context.h"
#include <vulkan/vulkan.h>

class NativeParticleBuffer {
    public:
        // Constructor and Destructor
        NativeParticleBuffer(size_t particle_count);
        ~NativeParticleBuffer();

        // Method to add particles to the buffer
        void upload(const Particle* particles, size_t count);

        // Getter for Vulkan buffer
        VkBuffer get_buffer() const { return vk_buffer; }
        // Getter for particle count
        size_t get_particle_count() const { return particle_count; }

    private:
        // Initialize Vulkan buffer to null handle
        VkBuffer vk_buffer = VK_NULL_HANDLE;
        // Initialize Vulkan device memory to null handle. We need this to reserve space in the device memory
        VkDeviceMemory vk_memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
};