// This file is a template file from Godot's GDNative module examples.
// It has been adapted for the life-particle-simulator plugin.

#include "register_types.h"
#include "cuda_particles_renderer.h"
#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot; // use Godot namespace for convenience

namespace godot {
    // Module initialization
    void initialize_gdextension_module(ModuleInitializationLevel p_level) {
        // Only initialize at the scene level
        if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
            return;
        }

        // Register the CudaParticlesRenderer class
        GDREGISTER_RUNTIME_CLASS(CudaParticlesRenderer);
    }

    // Module uninitialization
    void uninitialize_gdextension_module(ModuleInitializationLevel p_level) {
        // Only uninitialize at the scene level
        if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
            return;
        }
    }
}

// Entry point called by Godot to initialize the module
extern "C" {
// GDNative entry point
GDExtensionBool GDE_EXPORT cuda_particles_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address,
    const GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization *r_initialization
) {
    // Create the initialization object
	godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

    // Register the module's initialization and termination functions
	init_obj.register_initializer(godot::initialize_gdextension_module);
	init_obj.register_terminator(godot::uninitialize_gdextension_module);
	init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

    // Initialize the module
	return init_obj.init();
    }
}