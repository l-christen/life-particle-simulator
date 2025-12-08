#ifndef CUDA_PARTICLES_RENDERER_H
#define CUDA_PARTICLES_RENDERER_H

#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/classes/multi_mesh.hpp>
#include <cuda_runtime.h>
#include <vector>

#include "render_buffer.h"
#include "compute_buffers.h"
#include "simulation.cuh"

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
    
    protected:
        static void _bind_methods();
    
    public:
        CudaParticlesRenderer();
        ~CudaParticlesRenderer();

        void _ready() override;
        void _process(double delta) override;
        void CudaParticlesRenderer::update_multimesh(ParticlesAoS& render_buffer);
    };
}

#endif