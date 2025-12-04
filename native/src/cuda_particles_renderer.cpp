#include "cuda_particles_renderer.h"
#include <godot_cpp/core/class_db.hpp>

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

    // Initialize CUDA event
    cudaEventCreate(&simulation_done);

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
        compute->prev,
        compute->next,
        render->particlesAoS,
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
    multimesh->set_instance_count(render->num_particles);

    // Create mesh
    Ref<CircleMesh> mesh = memnew(CircleMesh);
    mesh->set_radius(2.0);
    mesh->set_sides(12);
    multimesh->set_mesh(mesh);

    // Create node to display the MultiMesh
    MultiMeshInstance2D* instance = memnew(MultiMeshInstance2D);
    instance->set_multimesh(multimesh);
    add_child(instance);
}

void CudaParticlesRenderer::_process(double delta) {
    // If simulation is not running, start a new step
    if (!simulation_running) {
        runSimulationStep(compute->prev, compute->next, render->particlesAoS, delta, simulation_done);
        simulation_running = true;
    }
    cudaError_t status = cudaEventQuery(simulation_done);
    if (status == cudaSuccess) {
        compute->swap();

        update_multimesh(render->particlesAoS);
        simulation_running = false;
    }
}

void CudaParticlesRenderer::update_multimesh(ParticlesAoS& render_buffer)
{
    int count = render_buffer.numParticles;
    Particle* particles = render_buffer.h_particles;

    // Transforms 2d
    PackedFloat32Array buffer;
    buffer.resize(count * 6); // 6 floats per transform (a, b, c, d, tx, ty)
    float *w = buffer.ptrw();

    for (int i = 0; i < count; i++) {
        int index = i * 6;

        w[index + 0] = 1.0f;               // a
        w[index + 1] = 0.0f;               // b
        w[index + 2] = 0.0f;               // c
        w[index + 3] = 1.0f;               // d
        w[index + 4] = particles[i].x;     // tx
        w[index + 5] = particles[i].y;     // ty
    }

    multimesh->set_buffer(buffer);

    // Colors per particle
    PackedColorArray colors;
    colors.resize(count);

    for (int i = 0; i < count; i++) {
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

        colors.set(i, col);
    }

    multimesh->set_color_array(colors);
}
