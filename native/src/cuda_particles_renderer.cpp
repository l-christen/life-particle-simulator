#include "cuda_particles_renderer.h"
#include "simulation.cuh"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/multi_mesh.hpp>
#include <godot_cpp/classes/multi_mesh_instance2d.hpp>
#include <godot_cpp/classes/quad_mesh.hpp>
#include <godot_cpp/classes/node2d.hpp>

using namespace godot;

void CudaParticlesRenderer::_bind_methods() {
}

CudaParticlesRenderer::CudaParticlesRenderer() {
}

CudaParticlesRenderer::~CudaParticlesRenderer() {
}

void CudaParticlesRenderer::_ready() {
    UtilityFunctions::print("CudaParticlesRenderer is ready!");
    
    // Initialize buffers with 1 million particles max capacity for now
    render = new RenderBuffer(1000000);
    compute = new ComputeBuffers(1000000);

    // Random rules
    float rules[25] = {
        0.0f,  0.5f, -0.3f,  0.2f, -0.4f,
        0.5f,  0.0f,  0.6f, -0.2f, -0.3f,
        -0.3f,  0.6f,  0.0f,  0.7f, -0.5f,
        0.2f, -0.2f,  0.7f,  0.0f,  0.4f,
        -0.4f, -0.3f, -0.5f,  0.4f,  0.0f
    };

    float radiusOfInfluence[5] = { 15.0f, 20.0f, 25.0f, 30.0f, 18.0f };

    // Initialize simulation
    initSimulation(
        &compute->prev,
        &compute->next,
        &render->particlesAoS,
        rules,
        radiusOfInfluence,
        1000,
        5,
        1024.0f,
        768.0f
    );

    // Initialize multimesh for rendering
    multimesh = memnew(MultiMesh);
    multimesh->set_use_colors(true);
    multimesh->set_transform_format(MultiMesh::TRANSFORM_2D);
    multimesh->set_instance_count(render->particlesAoS.numParticles);

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

void CudaParticlesRenderer::_process(double delta) {
    // If simulation is not running, start a new step
    UtilityFunctions::print("Processing CudaParticlesRenderer...");

    runSimulationStep(&compute->prev, &compute->next, &render->particlesAoS, render->particlesAoS.numParticles, 1024.0f, 768.0f, 0.5f);

    // Swap buffers for next iteration
    compute->swap();

    // Update multimesh with new particle positions
    update_multimesh(render->particlesAoS);

}

void CudaParticlesRenderer::update_multimesh(ParticlesAoS& render_buffer)
{
    int count = render_buffer.numParticles;
    Particle* particles = render_buffer.h_particles;

    UtilityFunctions::print("Updating MultiMesh with ", count, " particles.");

    // Update instance transforms and colors based on particle data
    for (int i = 0; i < count; i++) {
        Vector2 pos = Vector2(particles[i].x, particles[i].y);
        Transform2D transform;
        transform.set_origin(pos);
        multimesh->set_instance_transform_2d(i, transform);
        Color col;

        switch (particles[i].type) {
            case 1: // RED
                col = Color(1.0f, 0.2f, 0.2f, 1.0f);
                break;
            case 2: // BLUE
                col = Color(0.2f, 0.4f, 1.0f, 1.0f);
                break;
            case 3: // GREEN
                col = Color(0.2f, 1.0f, 0.4f, 1.0f);
                break;
            case 4: // YELLOW
                col = Color(1.0f, 1.0f, 0.3f, 1.0f);
                break;
            case 0:
            default:
                col = Color(0.0f, 0.0f, 0.0f, 0.0f);
                break;
        }

        multimesh->set_instance_color(i, col);
    }
}
