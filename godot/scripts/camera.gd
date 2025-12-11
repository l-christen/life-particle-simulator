extends Camera2D

const SPEED := 300.0
const ZOOM_STEP := 0.25
const UI_PANEL_WIDTH := 350.0

var sim_width: float = 0.0
var sim_height: float = 0.0

func _ready():
	var screen_width = DisplayServer.window_get_size().x
	var screen_height = DisplayServer.window_get_size().y
	
	sim_width = screen_width - UI_PANEL_WIDTH
	sim_height = screen_height
	
	var center_x = UI_PANEL_WIDTH + (sim_width / 2.0)
	var center_y = sim_height / 2.0
	
	# Camera centered in the middle of the simulation
	position = Vector2(center_x, center_y) 
	offset = Vector2(0, 0)

	# This limit are set because of the UI
	limit_left = int(UI_PANEL_WIDTH)
	limit_top = 0
	limit_right = int(screen_width) 
	limit_bottom = int(sim_height)

# Zoom method
func zoom():
	if Input.is_action_just_released('wheel_down'):
		set_zoom(get_zoom() - Vector2(ZOOM_STEP, ZOOM_STEP))
	if Input.is_action_just_released('wheel_up'):
		set_zoom(get_zoom() + Vector2(ZOOM_STEP, ZOOM_STEP))

# Method called each frame to move the camera
func _process(delta):
	var dir = Vector2.ZERO

	if Input.is_action_pressed("ui_up"):
		dir.y -= 1
	if Input.is_action_pressed("ui_down"):
		dir.y += 1
	if Input.is_action_pressed("ui_left"):
		dir.x -= 1
	if Input.is_action_pressed("ui_right"):
		dir.x += 1

	position += dir.normalized() * SPEED * delta
	
	position.x = clampf(position.x, limit_left, limit_right)
	position.y = clampf(position.y, limit_top, limit_bottom)
	
	zoom()
