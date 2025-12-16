# This file has been written with a lot of help from Gemini

extends Control

# Get Nodes when they are instantiates
@onready var particle_renderer = $"../..//CudaParticlesRenderer"

@onready var particle_count_panel = $GlobalVBox/ParticleCountHBox
@onready var rules_radius_panel = $GlobalVBox/RulesRadiusVBox

@onready var start_stop_button = $GlobalVBox/SimControlsHBox/StartStopButton
@onready var toggle_pause_button = $GlobalVBox/SimControlsHBox/TogglePauseButton
@onready var dt_viscosity_input_panel = $GlobalVBox/DtViscosityHbox

# 3 possibles simulation states
enum SimState { IDLE, RUNNING, PAUSED }
var current_state = SimState.IDLE

# Simulation dimensions declaration
var sim_width: int
var sim_height: int


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	# Initialize sim_height and sim_width
	sim_width = DisplayServer.window_get_size().x
	sim_height = DisplayServer.window_get_size().y
	
	# Setup sliders ranges and default values
	setup_slider_ranges()
	
	# Connect sliders and spinboxes to the eventlistener
	_connect_sliders()
	_connect_spinboxes()
	
	# Update UI components visibility
	update_ui_for_state()

	# Set position of the particle renderer
	particle_renderer.global_position = Vector2(0, 0)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta: float) -> void:
	pass

# Function to control components visibility according to simulation state
func update_ui_for_state():
	# Get input nodes for dt and viscosity to set their editability
	var dt_input_node = dt_viscosity_input_panel.get_node("DtInput")
	var viscosity_input_node = dt_viscosity_input_panel.get_node("ViscosityInput")

	# Update UI based on current state and set editability of dt and viscosity inputs
	match current_state:
		SimState.IDLE:
			particle_count_panel.visible = true
			rules_radius_panel.visible = true
			dt_viscosity_input_panel.visible = true
			toggle_pause_button.visible = false
			start_stop_button.visible = true
			start_stop_button.text = "Start"
			
			dt_input_node.editable = true
			viscosity_input_node.editable = true
			
		SimState.RUNNING:
			particle_count_panel.visible = false
			rules_radius_panel.visible = false
			dt_viscosity_input_panel.visible = false
			toggle_pause_button.visible = true
			toggle_pause_button.text = "Pause"
			start_stop_button.visible = true
			start_stop_button.text = "Stop"
			
			dt_input_node.editable = false
			viscosity_input_node.editable = false

			
		SimState.PAUSED:
			particle_count_panel.visible = false
			rules_radius_panel.visible = true
			dt_viscosity_input_panel.visible = true
			toggle_pause_button.visible = true
			toggle_pause_button.text = "Play"
			start_stop_button.visible = true
			start_stop_button.text = "Stop"
			
			dt_input_node.editable = true
			viscosity_input_node.editable = true
			
# Logic for the start stop button
func _on_start_stop_button_pressed():
	if current_state == SimState.IDLE:
		# Collect simulation parameters from the UI
		var params = collect_simulation_parameters()

		# Validate that at least one particle is present
		var total_particles = params.num_red + params.num_blue + params.num_green + params.num_yellow
		if total_particles == 0:
			print("Error: The number of particles has to be greater than 0.")
			return

		# Update simulation parameters in the renderer
		particle_renderer.update_delta_time(params.dt)
		particle_renderer.update_viscosity(params.viscosity)
		
		# Call the binded method from GDExtension
		particle_renderer.start_simulation(
			params.num_red,
			params.num_blue,
			params.num_green,
			params.num_yellow,
			params.rules,
			params.radius,
			sim_width,
			sim_height
			)
		
		# Modify simulation state and update UI
		current_state = SimState.RUNNING
		update_ui_for_state()
		
	elif current_state == SimState.RUNNING or current_state == SimState.PAUSED:
		particle_renderer.stop_simulation()
		
		# Modify simulation state and update UI
		current_state = SimState.IDLE
		update_ui_for_state()

# Logic for the play pause button
func _on_toggle_pause_button_pressed():
	if current_state == SimState.RUNNING:
		particle_renderer.update_is_running()
		current_state = SimState.PAUSED
		update_ui_for_state()
		
	elif current_state == SimState.PAUSED:
		particle_renderer.update_is_running()
		current_state = SimState.RUNNING
		update_ui_for_state()

# Get the numerical values from spinboxes or linedit
func get_numerical_input(path_to_input_node) -> float:
	var input_node = get_node(path_to_input_node)
	if input_node is SpinBox:
		return float(input_node.value)
	elif input_node is LineEdit:
		if input_node.text.is_valid_float():
			return float(input_node.text)
	return 0.0

# Get numerical values from sliders
func get_slider_rules() -> PackedFloat32Array:
	# Create an array to hold the rules values
	var rules_array = PackedFloat32Array()
	
	# Define the color order and target order
	# Order is important to have a consistent rules array
	var color_order = [
		{"key": "R", "vbox": $GlobalVBox/RulesRadiusVBox/RedContainer/RedParametersVBox},
		{"key": "B", "vbox": $GlobalVBox/RulesRadiusVBox/BlueContainer/BlueParametersVBox},
		{"key": "G", "vbox": $GlobalVBox/RulesRadiusVBox/GreenContainer/GreenParametersVBox},
		{"key": "Y", "vbox": $GlobalVBox/RulesRadiusVBox/YellowContainer/YellowParametersVBox}
	]
	var target_order = ["R", "B", "G", "Y"] 
	
	for source in color_order:
		var source_key = source.key # R, B, G, Y
		var source_vbox = source.vbox
		
		for target_key in target_order:
			# Construct the slider path
			var slider_container_name = source_key + "x" + target_key + "SliderHBox"
			
			var slider_path = slider_container_name + "/HSlider"
			
			var slider = source_vbox.get_node_or_null(slider_path)
			
			if slider and slider is HSlider:
				rules_array.append(slider.value)
			else:
				rules_array.append(0.0)
				
	return rules_array

# Get numerical values for the radius of influence
func get_radius_of_influence() -> PackedFloat32Array:
	# Create an array to hold the radius values
	var radius_array = PackedFloat32Array()
	
	# Define the color order
	# Order is important to have a consistent radius array
	var color_order = [
		{"name": "Red", "vbox": $GlobalVBox/RulesRadiusVBox/RedContainer/RedParametersVBox},
		{"name": "Blue", "vbox": $GlobalVBox/RulesRadiusVBox/BlueContainer/BlueParametersVBox},
		{"name": "Green", "vbox": $GlobalVBox/RulesRadiusVBox/GreenContainer/GreenParametersVBox},
		{"name": "Yellow", "vbox": $GlobalVBox/RulesRadiusVBox/YellowContainer/YellowParametersVBox}
	]
	
	for color in color_order:
		var color_name = color.name
		var color_vbox = color.vbox
		
		# Construct the input path
		var radius_container_name = color_name + "RadiusHBox"
		var input_path = radius_container_name + "/RadiusInput"
		
		var input_node = color_vbox.get_node_or_null(input_path)
		
		if input_node:
			var radius_value = get_numerical_input(input_node.get_path())
			radius_array.append(radius_value)
		else:
			radius_array.append(0.0)
			print("Error: Node not found for %s." % color_name)
			
	return radius_array

# Main function to collect all parameters
func collect_simulation_parameters() -> Dictionary:
	# Get particle counts from the UI
	var num_red = get_numerical_input(str(particle_count_panel.get_path()) + "/RedParticleCount/RedCountInput")
	var num_blue = get_numerical_input(str(particle_count_panel.get_path()) + "/BlueParticleCount/BlueCountInput")
	var num_green = get_numerical_input(str(particle_count_panel.get_path()) + "/GreenParticleCount/GreenCountInput")
	var num_yellow = get_numerical_input(str(particle_count_panel.get_path()) + "/YellowParticleCount/YellowCountInput")
	
	# Get dt and viscosity values from the UI
	var dt_value = get_numerical_input(str(dt_viscosity_input_panel.get_path()) + "/DtInput")
	var viscosity_value = get_numerical_input(str(dt_viscosity_input_panel.get_path()) + "/ViscosityInput")

	# Get rules and radius arrays
	var rules_array = get_slider_rules()
	var radius_array = get_radius_of_influence()

	# Return all parameters in a dictionary
	return {
		"num_red": int(num_red),
		"num_blue": int(num_blue),
		"num_green": int(num_green),
		"num_yellow": int(num_yellow),
		"rules": rules_array,
		"radius": radius_array,
		"viscosity": viscosity_value,
		"dt": dt_value
	}

# Method called when a value change in the UI (main eventlistener)
func _on_parameter_input_changed(_new_value):
	if current_state == SimState.PAUSED:
		# Collect updated parameters if the simulation is paused
		var rules_array = get_slider_rules()
		var radius_array = get_radius_of_influence()
		var dt_value = get_numerical_input(str(dt_viscosity_input_panel.get_path()) + "/DtInput")
		var viscosity_value = get_numerical_input(str(dt_viscosity_input_panel.get_path()) + "/ViscosityInput")
		
		# Update the simulation parameters in the renderer
		particle_renderer.update_rules(rules_array)
		particle_renderer.update_radius_of_influence(radius_array)
		particle_renderer.update_delta_time(dt_value)
		particle_renderer.update_viscosity(viscosity_value)
		

# Method called when a slider value change to modify the value in the slider label
func _on_h_slider_value_changed(value: float, slider_node: HSlider) -> void:
	# Update the corresponding label next to the slider
	if slider_node and slider_node is HSlider:
		var value_label = slider_node.get_node("../SliderValue")
		if value_label and value_label is Label:
			# Update label text with formatted value
			value_label.text = "%.2f" % value
			
			# If simulation is paused, update the simulation parameters
			if current_state == SimState.PAUSED:
				_on_parameter_input_changed(value)

# Method to setup sliders on _ready
func setup_slider_ranges():
	const MIN_RULE_VALUE: float = -2000.0
	const MAX_RULE_VALUE: float = 2000.0
	const SLIDER_STEP: float = 2.0
	
	# Define the container nodes for each color
	var color_vboxes = [
		$GlobalVBox/RulesRadiusVBox/RedContainer/RedParametersVBox,
		$GlobalVBox/RulesRadiusVBox/BlueContainer/BlueParametersVBox,
		$GlobalVBox/RulesRadiusVBox/GreenContainer/GreenParametersVBox,
		$GlobalVBox/RulesRadiusVBox/YellowContainer/YellowParametersVBox
	]
	
	# Iterate through all color VBoxes to setup sliders
	for color_vbox in color_vboxes:
		for child in color_vbox.get_children():
			
			var slider = child.get_node_or_null("HSlider")
			
			if slider and slider is HSlider:
				slider.min_value = MIN_RULE_VALUE
				slider.max_value = MAX_RULE_VALUE
				slider.step = SLIDER_STEP
				slider.value = 0.0

# Method to connect sliders to their eventlistener
func _connect_sliders():
	# Iterate through all color VBoxes to connect sliders
	var color_vboxes = [
		$GlobalVBox/RulesRadiusVBox/RedContainer/RedParametersVBox,
		$GlobalVBox/RulesRadiusVBox/BlueContainer/BlueParametersVBox,
		$GlobalVBox/RulesRadiusVBox/GreenContainer/GreenParametersVBox,
		$GlobalVBox/RulesRadiusVBox/YellowContainer/YellowParametersVBox
	]
	
	# Connect slider signals
	for color_vbox in color_vboxes:
		for child in color_vbox.get_children():
			var slider = child.get_node_or_null("HSlider")
			
			if slider and slider is HSlider:
				# Create a callable with the slider bound as an argument
				var callable = Callable(self, "_on_h_slider_value_changed").bind(slider)
				if not slider.is_connected("value_changed", callable):
					slider.connect("value_changed", callable)

# Method to connect spinboxes to their eventlistener (useful to modify simulation parameters when state is PAUSED)
func _connect_spinboxes():
	# Define radius spinboxes
	var radius_vboxes = [
		$GlobalVBox/RulesRadiusVBox/RedContainer/RedParametersVBox/RedRadiusHBox,
		$GlobalVBox/RulesRadiusVBox/BlueContainer/BlueParametersVBox/BlueRadiusHBox,
		$GlobalVBox/RulesRadiusVBox/GreenContainer/GreenParametersVBox/GreenRadiusHBox,
		$GlobalVBox/RulesRadiusVBox/YellowContainer/YellowParametersVBox/YellowRadiusHBox
	]
	# Iterate through radius spinboxes
	for vbox in radius_vboxes:
		var input_node = vbox.get_node_or_null("RadiusInput")
		if input_node and input_node is SpinBox:
			# Create a callable with the input_node bound as an argument
			var callable = Callable(self, "_on_parameter_input_changed").bind(input_node)
			if not input_node.is_connected("value_changed", callable):
				input_node.connect("value_changed", callable)

	# Connect dt and viscosity spinboxes
	var dt_node = dt_viscosity_input_panel.get_node_or_null("DtInput")
	if dt_node and dt_node is SpinBox:
		# Create a callable with the dt_node bound as an argument
		var callable = Callable(self, "_on_parameter_input_changed").bind(dt_node)
		if not dt_node.is_connected("value_changed", callable):
			dt_node.connect("value_changed", callable)
	var viscosity_node = dt_viscosity_input_panel.get_node_or_null("ViscosityInput")
	if viscosity_node and viscosity_node is SpinBox:
		# Create a callable with the viscosity_node bound as an argument
		var callable = Callable(self, "_on_parameter_input_changed").bind(viscosity_node)
		if not viscosity_node.is_connected("value_changed", callable):
			viscosity_node.connect("value_changed", callable)