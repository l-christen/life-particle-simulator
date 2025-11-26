#ifndef CUDA_PARTICLES_RENDERER_H
#define CUDA_PARTICLES_RENDERER_H

#include <godot_cpp/classes/node2d.hpp>

namespace godot {
    class CudaParticlesRenderer : public Node2D {
        GDCLASS(CudaParticlesRenderer, Node2D)
    
    protected:
        static void _bind_methods();
    
    public:
        CudaParticlesRenderer();
        ~CudaParticlesRenderer();

        void _ready() override;
        void _process(double delta) override;
    };
}

#endif