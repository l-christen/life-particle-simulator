#ifndef CUDA_PARTICLES_RENDERER_H
#define CUDA_PARTICLES_RENDERER_H

#include <godot_cpp/classes/node2d.hpp>
#include <cuda_runtime.h>

#include "render_buffer.h"
#include "compute_buffers.h"

namespace godot {
    class CudaParticlesRenderer : public Node2D {
        GDCLASS(CudaParticlesRenderer, Node2D)
    
    private:
        // Simulation and rendering buffers
        RenderBuffer* render = nullptr;
        ComputeBuffers* compute = nullptr;

        // Multimesh instance for rendering particles
        MultiMesh* multimesh;
        RID multimesh_rid;
        std::vector<float> transform_buffer;

        // Simulation control
        bool simulation_running = false;
        // CUDA event to signal simulation completion, this should allow Godot to render while
        // the simulation is running without waiting for a step to complete
        // proposed by ChatGPT, has to be tested properly
        cudaEvent_t simulation_done;
    
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