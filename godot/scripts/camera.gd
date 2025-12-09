extends Camera2D

const SPEED := 300.0
const ZOOM_STEP := 0.25
const UI_PANEL_WIDTH := 350.0

var sim_width: float = 0.0
var sim_height: float = 0.0

func _ready():
	sim_width = DisplayServer.window_get_size().x - UI_PANEL_WIDTH
	sim_height = DisplayServer.window_get_size().y
	
	position = Vector2(sim_width / 2.0, sim_height / 2.0)

	limit_left = 0
	limit_top = 0
	limit_right = int(sim_width)
	limit_bottom = int(sim_height)
	
	offset = Vector2(UI_PANEL_WIDTH / 2.0, 0)

func zoom():
	if Input.is_action_just_released('wheel_down'):
		set_zoom(get_zoom() - Vector2(ZOOM_STEP, ZOOM_STEP))
	if Input.is_action_just_released('wheel_up'):
		set_zoom(get_zoom() + Vector2(ZOOM_STEP, ZOOM_STEP))

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
