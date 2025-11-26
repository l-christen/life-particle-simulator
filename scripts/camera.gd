extends Camera2D

const SPEED := 300

# https://forum.godotengine.org/t/how-do-i-change-camera-zoom-using-a-script/27630/4
func zoom():
	if Input.is_action_just_released('wheel_down'):
		set_zoom(get_zoom() - Vector2(0.25, 0.25))
	if Input.is_action_just_released('wheel_up'): #and get_zoom() > Vector2.ONE:
		set_zoom(get_zoom() + Vector2(0.25, 0.25))

# made with chatgpt
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
	
	zoom()
