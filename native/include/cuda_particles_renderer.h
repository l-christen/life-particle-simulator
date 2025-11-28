#ifndef CUDA_PARTICLES_RENDERER_H
#define CUDA_PARTICLES_RENDERER_H

#include "native_particle_buffer.h"

#include <godot_cpp/classes/node2d.hpp>
#include <vulkan/vulkan.h>

namespace godot {
    class CudaParticlesRenderer : public Node2D {
        // Declare the class to Godot's type system
        GDCLASS(CudaParticlesRenderer, Node2D)
    
    protected:
        static void _bind_methods();
    
    public:
        CudaParticlesRenderer();
        ~CudaParticlesRenderer();
        
        void _ready() override;
        void _process(double delta) override;

    private:
        bool buffer_initialized = false; // flag to check if buffer is initialized
        std::unique_ptr<NativeParticleBuffer> native_buffer; // unique pointer to native particle buffer
        std::vector<Particle> particles_cpu; // CPU-side particle data
    };
}

#endif