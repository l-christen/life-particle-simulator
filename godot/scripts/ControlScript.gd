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

var sim_width: int
var sim_height: int


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	# Initialize sim_height and sim_width
	sim_width = DisplayServer.window_get_size().x
	sim_height = DisplayServer.window_get_size().y
	
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
	var dt_input_node = dt_viscosity_input_panel.get_node("DtInput")
	var viscosity_input_node = dt_viscosity_input_panel.get_node("ViscosityInput")

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
		var params = collect_simulation_parameters()
		# Debug print
		print(params)

		var total_particles = params.num_red + params.num_blue + params.num_green + params.num_yellow
		if total_particles == 0:
			print("Error: The number of particles has to be greater than 0.")
			return
		
		# Call the binded method from GDExtension
		particle_renderer.start_simulation(
			params.num_red,
			params.num_blue,
			params.num_green,
			params.num_yellow,
			params.rules,
			params.radius,
			sim_width,
			sim_height,
			params.dt
			)
		print("Sim started")
		
		# Modify simulation state
		current_state = SimState.RUNNING
		update_ui_for_state()
		
	elif current_state == SimState.RUNNING or current_state == SimState.PAUSED:
		particle_renderer.stop_simulation()
		
		current_state = SimState.IDLE
		update_ui_for_state()

# Logig for the play pause button
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
	var rules_array = PackedFloat32Array()
	
	var color_data = [
		{"key": "R", "vbox": $GlobalVBox/RulesRadiusVBox/RedContainer/RedParametersVBox},
		{"key": "B", "vbox": $GlobalVBox/RulesRadiusVBox/BlueContainer/BlueParametersVBox},
		{"key": "G", "vbox": $GlobalVBox/RulesRadiusVBox/GreenContainer/GreenParametersVBox},
		{"key": "Y", "vbox": $GlobalVBox/RulesRadiusVBox/YellowContainer/YellowParametersVBox}
	]
	var target_order = ["R", "B", "G", "Y"] 
	
	for source in color_data:
		var source_key = source.key # R, B, G, ou Y
		var source_vbox = source.vbox
		
		for target_key in target_order:
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
	var radius_array = PackedFloat32Array()
	
	var color_order = [
		{"name": "Red", "vbox": $GlobalVBox/RulesRadiusVBox/RedContainer/RedParametersVBox},
		{"name": "Blue", "vbox": $GlobalVBox/RulesRadiusVBox/BlueContainer/BlueParametersVBox},
		{"name": "Green", "vbox": $GlobalVBox/RulesRadiusVBox/GreenContainer/GreenParametersVBox},
		{"name": "Yellow", "vbox": $GlobalVBox/RulesRadiusVBox/YellowContainer/YellowParametersVBox}
	]
	
	for color in color_order:
		var color_name = color.name
		var color_vbox = color.vbox
		
		var radius_container_name = color_name + "RadiusHBox"
		var input_path = radius_container_name + "/RadiusInput"
		
		var input_node = color_vbox.get_node_or_null(input_path)
		
		if input_node:
			var radius_value = get_numerical_input(input_node.get_path())
			radius_array.append(radius_value)
		else:
			radius_array.append(0.0)
			print("Erreur: Node not found for %s." % color_name)
			
	return radius_array

# Main function to collect all parameters
func collect_simulation_parameters() -> Dictionary:
	var num_red = get_numerical_input(str(particle_count_panel.get_path()) + "/RedParticleCount/RedCountInput")
	var num_blue = get_numerical_input(str(particle_count_panel.get_path()) + "/BlueParticleCount/BlueCountInput")
	var num_green = get_numerical_input(str(particle_count_panel.get_path()) + "/GreenParticleCount/GreenCountInput")
	var num_yellow = get_numerical_input(str(particle_count_panel.get_path()) + "/YellowParticleCount/YellowCountInput")
	
	var dt_value = get_numerical_input(str(dt_viscosity_input_panel.get_path()) + "/DtInput")
	var viscosity_value = get_numerical_input(str(dt_viscosity_input_panel.get_path()) + "/ViscosityInput")

	var rules_array = get_slider_rules()
	var radius_array = get_radius_of_influence()

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

# Method called when a value change in the UI
func _on_parameter_input_changed(_new_value):
	if current_state == SimState.PAUSED:
		var rules_array = get_slider_rules()
		var radius_array = get_radius_of_influence()
		var dt_value = get_numerical_input(str(dt_viscosity_input_panel.get_path()) + "/DtInput")
		var viscosity_value = get_numerical_input(str(dt_viscosity_input_panel.get_path()) + "/ViscosityInput")
		
		particle_renderer.update_rules(rules_array)
		particle_renderer.update_radius_of_influence(radius_array)
		particle_renderer.update_delta_time(dt_value)
		particle_renderer.update_viscosity(viscosity_value)
		

# Method called when a slider value change to modify the value in the slider label
func _on_h_slider_value_changed(value: float, slider_node: HSlider) -> void:
	if slider_node and slider_node is HSlider:
		var value_label = slider_node.get_node("../SliderValue")
		if value_label and value_label is Label:
			value_label.text = "%.2f" % value
			
			if current_state == SimState.PAUSED:
				_on_parameter_input_changed(value)
		value_label.text = "%.2f" % value

# Method to setup sliders on _ready
func setup_slider_ranges():
	const MIN_RULE_VALUE: float = -20.0
	const MAX_RULE_VALUE: float = 20.0
	const SLIDER_STEP: float = 0.1
	
	var color_vboxes = [
		$GlobalVBox/RulesRadiusVBox/RedContainer/RedParametersVBox,
		$GlobalVBox/RulesRadiusVBox/BlueContainer/BlueParametersVBox,
		$GlobalVBox/RulesRadiusVBox/GreenContainer/GreenParametersVBox,
		$GlobalVBox/RulesRadiusVBox/YellowContainer/YellowParametersVBox
	]
	
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
	var color_vboxes = [
		$GlobalVBox/RulesRadiusVBox/RedContainer/RedParametersVBox,
		$GlobalVBox/RulesRadiusVBox/BlueContainer/BlueParametersVBox,
		$GlobalVBox/RulesRadiusVBox/GreenContainer/GreenParametersVBox,
		$GlobalVBox/RulesRadiusVBox/YellowContainer/YellowParametersVBox
	]
	
	for color_vbox in color_vboxes:
		for child in color_vbox.get_children():
			var slider = child.get_node_or_null("HSlider")
			
			if slider and slider is HSlider:
				var callable = Callable(self, "_on_h_slider_value_changed").bind(slider)
				if not slider.is_connected("value_changed", callable):
					slider.connect("value_changed", callable)

# Method to connect spinboxes to their eventlistener (usefull to modify simulation parameters when state is PAUSED)
func _connect_spinboxes():
	var radius_vboxes = [
		$GlobalVBox/RulesRadiusVBox/RedContainer/RedParametersVBox,
		$GlobalVBox/RulesRadiusVBox/BlueContainer/BlueParametersVBox,
		$GlobalVBox/RulesRadiusVBox/GreenContainer/GreenParametersVBox,
		$GlobalVBox/RulesRadiusVBox/YellowContainer/YellowParametersVBox
	]
	for vbox in radius_vboxes:
		var input_node = vbox.get_node_or_null("RedRadiusHBox/RadiusInput")
		if input_node and input_node is SpinBox:
			if not input_node.is_connected("value_changed", Callable(self, "_on_parameter_input_changed")):
				input_node.connect("value_changed", Callable(self, "_on_parameter_input_changed"))

	var count_nodes = [
		$GlobalVBox/ParticleCountHBox/RedParticleCount/RedCountInput,
		$GlobalVBox/ParticleCountHBox/BlueParticleCount/BlueCountInput,
		$GlobalVBox/ParticleCountHBox/GreenParticleCount/GreenCountInput,
		$GlobalVBox/ParticleCountHBox/YellowParticleCount/YellowCountInput,
	]
	for node in count_nodes:
		if node and node is SpinBox:
			if not node.is_connected("value_changed", Callable(self, "_on_count_input_changed")):
				node.connect("value_changed", Callable(self, "_on_count_input_changed"))

	var dt_node = dt_viscosity_input_panel.get_node_or_null("DtInput")
	if dt_node and dt_node is SpinBox:
		if not dt_node.is_connected("value_changed", Callable(self, "_on_parameter_input_changed")):
			dt_node.connect("value_changed", Callable(self, "_on_parameter_input_changed"))

	var viscosity_node = dt_viscosity_input_panel.get_node_or_null("ViscosityInput")
	if viscosity_node and viscosity_node is SpinBox:
		if not viscosity_node.is_connected("value_changed", Callable(self, "_on_parameter_input_changed")):
			viscosity_node.connect("value_changed", Callable(self, "_on_parameter_input_changed"))
			
func _on_count_input_changed(_new_value: float):
	pass
