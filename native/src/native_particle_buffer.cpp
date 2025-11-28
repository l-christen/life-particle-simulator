#include "native_particle_buffer.h"
#include <cstring>

// This file has mainly be written by ChatGpt

// Helper function to find suitable memory type
static uint32_t find_memory_type(uint32_t typeFilter,
                                 VkMemoryPropertyFlags properties,
                                 VkPhysicalDevice physical) {
    // Get memory properties of the physical device                                
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical, &mem_props);
    
    // Find suitable memory type
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((typeFilter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0;
}

// Constructor
NativeParticleBuffer::NativeParticleBuffer(size_t particle_count) {
    // Get Vulkan context to access device and physical device
    VkContext &context = VkContext::get();
    VkDevice device = context.get_device();
    VkPhysicalDevice physical = context.get_physical_device();

    // Calculate buffer size
    size = static_cast<VkDeviceSize>(particle_count * sizeof(Particle));

    // To create Vulkan buffer we need to to set up VkBufferCreateInfo structure
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; // type of structure
    info.size = size; // size of the buffer in bytes
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; // source for copying to Godot buffer
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // not shared between multiple queue families

    // Create the Vulkan buffer
    vkCreateBuffer(device, &info, nullptr, &buffer);

    // Request memory requirements for the buffer
    VkMemoryRequirements mem_req;
    // Get memory requirements for the buffer
    vkGetBufferMemoryRequirements(device, buffer, &mem_req);

    // Find suitable memory type index
    uint32_t type_index = find_memory_type(
        mem_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        physical
    );

    // Allocate memory for the buffer
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = mem_req.size;
    alloc.memoryTypeIndex = type_index;
    
    // Allocate memory for the buffer
    vkAllocateMemory(device, &alloc, nullptr, &memory);
    // Bind the allocated memory to the buffer
    vkBindBufferMemory(device, buffer, memory, 0);
}

// Destructor
NativeParticleBuffer::~NativeParticleBuffer() {
    VkContext &context = VkContext::get();
    VkDevice device = context.get_device();

    // Free the allocated memory and destroy the buffer
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
    }
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
    }
}

// Method to upload particle data to the Vulkan buffer
void NativeParticleBuffer::upload(const Particle* data, size_t count) {
    // Get Vulkan context to access device
    VkContext &context = VkContext::get();
    VkDevice device = context.get_device();

    // Calculate size to upload
    VkDeviceSize upload_size = static_cast<VkDeviceSize>(count * sizeof(Particle));
    if (upload_size > size) {
        UtilityFunctions::print("Upload size exceeds buffer size!");
        upload_size = size;
    }

    // Map memory and copy data
    void* mapped = nullptr;
    vkMapMemory(device, memory, 0, upload_size, 0, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(upload_size));
    vkUnmapMemory(device, memory);
}