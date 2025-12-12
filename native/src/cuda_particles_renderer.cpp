#include "cuda_particles_renderer.h"


#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/multi_mesh.hpp>
#include <godot_cpp/classes/multi_mesh_instance2d.hpp>
#include <godot_cpp/classes/quad_mesh.hpp>
#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/core/property_info.hpp>

#include "simulation.cuh"
#include "compute_buffers.h"


using namespace godot;

// Binding methods for GDExtension, only static methods can be bound
void CudaParticlesRenderer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("start_simulation", "numRed", "numBlue", "numGreen", "numYellow", "simulationRules", "simulationRadiusOfInfluence", "width", "height", "deltaTime"), 
        &CudaParticlesRenderer::start_simulation);
    
    ClassDB::bind_method(D_METHOD("update_rules", "simulationRules"), 
        &CudaParticlesRenderer::update_rules);
    
    ClassDB::bind_method(D_METHOD("update_radius_of_influence", "simulationRadiusOfInfluence"), 
        &CudaParticlesRenderer::update_radius_of_influence);
    
    ClassDB::bind_method(D_METHOD("update_is_running"), 
        &CudaParticlesRenderer::update_is_running);
    
    ClassDB::bind_method(D_METHOD("stop_simulation"), 
        &CudaParticlesRenderer::stop_simulation);

    ClassDB::bind_method(D_METHOD("update_delta_time", "p_delta_time"), 
        &CudaParticlesRenderer::update_delta_time);

    ClassDB::bind_method(D_METHOD("update_viscosity", "p_viscosity"), 
        &CudaParticlesRenderer::update_viscosity);
}

CudaParticlesRenderer::CudaParticlesRenderer() {
}

CudaParticlesRenderer::~CudaParticlesRenderer() {
    // Free compute buffers
    if (compute) {
        delete compute;
    }
}

// Called when the node is added to the scene
void CudaParticlesRenderer::_ready() {
    UtilityFunctions::print("CudaParticlesRenderer is ready!");

    // Initialize buffers with 1 million particles max capacity for now
    compute = new ComputeBuffers(1000000);

    // Initialize multimesh for rendering
    multimesh = memnew(MultiMesh);
    multimesh->set_use_colors(true);
    multimesh->set_transform_format(MultiMesh::TRANSFORM_2D);

    // Create mesh
    godot::Ref<godot::QuadMesh> mesh;
    mesh.instantiate();
    mesh->set_size(godot::Vector2(2.0f, 2.0f));
    multimesh->set_mesh(mesh);

    // Create node to display the MultiMesh
    MultiMeshInstance2D* instance = memnew(MultiMeshInstance2D);
    instance->set_multimesh(multimesh);
    add_child(instance);
}

// Called every frame
void CudaParticlesRenderer::_process(double delta) {
    if (!is_initialized) {
        return;
    }
    if (is_initialized && is_running) {
        // Run simulation step
        runSimulationStep(&compute->prev, &compute->next, &compute->renderBuffer, num_particles, sim_width, sim_height, viscosity, delta_time);
    
        // Swap buffers for next iteration
        compute->swap();

        // Update multimesh with new particle positions
        update_multimesh(compute->renderBuffer);
    }
}

// Update the multimesh instance with the latest particle data
void CudaParticlesRenderer::update_multimesh(ParticlesAoS& render_buffer)
{
    int count = render_buffer.numParticles;
    Particle* particles = render_buffer.h_particles;

    // Update instance transforms and colors based on particle data
    // Can be optimized later by pushing tranforms/colors in one call
    for (int i = 0; i < count; i++) {
        Vector2 pos = Vector2(particles[i].x, particles[i].y);
        Transform2D transform;
        transform.set_origin(pos);
        multimesh->set_instance_transform_2d(i, transform);
        Color col;
        switch (particles[i].type) {
            case 0: // RED
                col = Color(1.0f, 0.2f, 0.2f, 1.0f);
                break;
            case 1: // BLUE
                col = Color(0.2f, 0.4f, 1.0f, 1.0f);
                break;
            case 2: // GREEN
                col = Color(0.2f, 1.0f, 0.4f, 1.0f);
                break;
            case 3: // YELLOW
                col = Color(1.0f, 1.0f, 0.3f, 1.0f);
                break;
            default:
                col = Color(0.0f, 0.0f, 0.0f, 0.0f);
                break;
        }

        multimesh->set_instance_color(i, col);
    }
}

// Start the simulation with given parameters
void CudaParticlesRenderer::start_simulation(
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
) {
    UtilityFunctions::print("Starting simulation");
    // Set the number of particles
    num_particles = numRed + numBlue + numGreen + numYellow;

    // Set sim_width, sim_height, delta_time
    sim_width = static_cast<float>(width);
    sim_height = static_cast<float>(height);
    delta_time = deltaTime;
    this->viscosity = viscosity;

    // Update rules and radius of influence
    update_rules(simulationRules);
    update_radius_of_influence(simulationRadiusOfInfluence);

    initSimulation(
        &compute->prev,
        &compute->next,
        &compute->renderBuffer,
        numRed,
        numBlue,
        numGreen,
        numYellow,
        sim_width,
        sim_height
    );

    // Set the number of particles in the multimesh
    multimesh->set_instance_count(num_particles);

    is_initialized = true;
    is_running = true;
}


// Update simulation rules
void CudaParticlesRenderer::update_rules(PackedFloat32Array simulationRules) {
    // Get raw pointer to the rules data, allowed since PackedFloat32Array stores data contiguously
    const float* raw_rules_ptr = simulationRules.ptr();

    // Update rules in device constant memory
    setSimulationRules(raw_rules_ptr);
}

// Update simulation radius of influence
void CudaParticlesRenderer::update_radius_of_influence(PackedFloat32Array simulationRadiusOfInfluence) {
    // Get raw pointer to the radius of influence data
    const float* raw_radius_ptr = simulationRadiusOfInfluence.ptr();

    // Update radius of influence in device constant memory
    setSimulationRadiusOfInfluence(raw_radius_ptr);
}

// Toggle simulation running state
void CudaParticlesRenderer::update_is_running() {
    is_running = !is_running;
}

// Stop the simulation
void CudaParticlesRenderer::stop_simulation() {
    is_running = false;
    is_initialized = false;
}
