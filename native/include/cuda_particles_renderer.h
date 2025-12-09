#ifndef CUDA_PARTICLES_RENDERER_H
#define CUDA_PARTICLES_RENDERER_H

#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/classes/multi_mesh.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <cuda_runtime.h>
#include <vector>

#include "compute_buffers.h"
#include "simulation.cuh"

namespace godot {
    class CudaParticlesRenderer : public Node2D {
        GDCLASS(CudaParticlesRenderer, Node2D)
    
    private:
        // Simulation and rendering buffers
        ComputeBuffers* compute = nullptr;

        // numParticles for current simulation
        uint32_t num_particles = 0;

        // Width and height of the simulation area
        float sim_width = 1024.0f;
        float sim_height = 768.0f;

        // State flags
        bool is_initialized = false;
        bool is_running = false;

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

        // Delta time for simulation steps
        float delta_time = 0.01f;

        void _ready() override;
        void _process(double delta) override;
        void CudaParticlesRenderer::update_multimesh(ParticlesAoS& render_buffer);
        void CudaParticlesRenderer::start_simulation(
            int numRed,
            int numBlue,
            int numGreen,
            int numYellow,
            PackedFloat32Array simulationRules,
            PackedFloat32Array simulationRadiusOfInfluence,
            int width,
            int height,
            float deltaTime
        );
        void CudaParticlesRenderer::update_rules(PackedFloat32Array simulationRules);
        void CudaParticlesRenderer::update_radius_of_influence(PackedFloat32Array simulationRadiusOfInfluence);
        void CudaParticlesRenderer::update_is_running();
        void CudaParticlesRenderer::stop_simulation();
    };
}

#endif