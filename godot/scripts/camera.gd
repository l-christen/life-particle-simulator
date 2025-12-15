extends Camera2D

const ZOOM_STEP := 0.25

func _ready():
	var screen_width = DisplayServer.window_get_size().x
	var screen_height = DisplayServer.window_get_size().y
	
	var center_x = screen_width / 2.0
	var center_y = screen_height / 2.0
	
	# Camera centered in the middle of the screen
	position = Vector2(center_x, center_y) 
	offset = Vector2(0, 0)

# Zoom method
func zoom():
	if Input.is_action_just_released('wheel_down'):
		set_zoom(get_zoom() - Vector2(ZOOM_STEP, ZOOM_STEP))
	if Input.is_action_just_released('wheel_up'):
		set_zoom(get_zoom() + Vector2(ZOOM_STEP, ZOOM_STEP))

# Method called each frame to zoom the camera
func _process(delta):
	zoom()
