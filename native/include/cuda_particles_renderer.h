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
        // Binding methods for GDExtension exposure, only way found to bind methods
        static void _start_simulation_bind(CudaParticlesRenderer* obj, int numRed, int numBlue, int numGreen, int numYellow, 
                                        PackedFloat32Array simulationRules, PackedFloat32Array simulationRadiusOfInfluence, 
                                        int width, int height, float deltaTime) {
            obj->start_simulation(numRed, numBlue, numGreen, numYellow, simulationRules, simulationRadiusOfInfluence, width, height, deltaTime);
        }
        
        static void _update_rules_bind(CudaParticlesRenderer* obj, PackedFloat32Array simulationRules) {
            obj->update_rules(simulationRules);
        }
        
        static void _update_radius_of_influence_bind(CudaParticlesRenderer* obj, PackedFloat32Array simulationRadiusOfInfluence) {
            obj->update_radius_of_influence(simulationRadiusOfInfluence);
        }
        
        static void _update_is_running_bind(CudaParticlesRenderer* obj) {
            obj->update_is_running();
        }
        
        static void _stop_simulation_bind(CudaParticlesRenderer* obj) {
            obj->stop_simulation();
        }
        
        static void _update_delta_time_bind(CudaParticlesRenderer* obj, float delta_time) {
            obj->update_delta_time(delta_time);
        }

        // Simulation and rendering buffers
        ComputeBuffers* compute = nullptr;

        // NumParticles for current simulation
        uint32_t num_particles = 0;

        // Delta time for simulation steps
        float delta_time = 0.01f;

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
            float deltaTime
        );
        void update_rules(PackedFloat32Array simulationRules);
        void update_radius_of_influence(PackedFloat32Array simulationRadiusOfInfluence);
        void update_is_running();
        void stop_simulation();
    };
}

#endif