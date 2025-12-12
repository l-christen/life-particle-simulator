#ifndef CUDA_PARTICLES_RENDERER_H
#define CUDA_PARTICLES_RENDERER_H

#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/classes/multi_mesh.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <vector>

class ComputeBuffers;
struct ParticlesAoS;

namespace godot {
    class CudaParticlesRenderer : public Node2D {
        GDCLASS(CudaParticlesRenderer, Node2D)
    
    private:
        // Simulation and rendering buffers
        ComputeBuffers* compute = nullptr;

        // NumParticles for current simulation
        uint32_t num_particles = 0;

        // Delta time for simulation steps
        float delta_time = 0.01f;

        // Viscosity parameter for simulation
        float viscosity = 0.0f;

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

        // Setter for delta_time
        void update_delta_time(float p_delta_time) { delta_time = p_delta_time; }

        // Setter for viscosity
        void update_viscosity(float p_viscosity) { viscosity = p_viscosity; }

        // Standard Godot methods
        void _ready() override;
        void _process(double delta) override;

        // Update the multimesh instance with the latest particle data
        void update_multimesh(ParticlesAoS& render_buffer);

        // Simulation control methods
        void start_simulation(
            int numRed,
            int numBlue,
            int numGreen,
            int numYellow,
            PackedFloat32Array simulationRules,
            PackedFloat32Array simulationRadiusOfInfluence,
            int width,
            int height,
            float viscosity,
            float deltaTime
        );
        void update_rules(PackedFloat32Array simulationRules);
        void update_radius_of_influence(PackedFloat32Array simulationRadiusOfInfluence);
        void update_is_running();
        void stop_simulation();
    };
}

#endif