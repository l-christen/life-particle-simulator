extends MultiMeshInstance2D

const PARTICLE_COUNT := 10
var velocities := []

# place holder code made with chatgpt, first steps with godot
func _ready() -> void:
	var mm := MultiMesh.new()
	mm.use_colors = true
	mm.transform_format = MultiMesh.TRANSFORM_2D
	mm.instance_count = PARTICLE_COUNT
	
	multimesh = mm
	
	var mesh_tex := MeshInstance2D.new()
	var quad := QuadMesh.new()
	quad.size = Vector2(4, 4)
	mesh_tex.mesh = quad
	mm.mesh = quad
	
	for i in range(PARTICLE_COUNT):
		var pos := Vector2(
			randi() % 800,
			randi() % 600
		)
		var transform := Transform2D(0.0, pos)
		mm.set_instance_transform_2d(i, transform)

		var r = randf()
		var g = randf()
		var b = randf()
		mm.set_instance_color(i, Color(r, g, b, 1.0))
		velocities.append(Vector2(randf_range(-2, 2), randf_range(-2, 2)))

func _process(delta: float) -> void:
	for i in range(PARTICLE_COUNT):
		var current_transform := multimesh.get_instance_transform_2d(i)
		var pos := current_transform.origin
		pos += velocities[i]
		current_transform.origin = pos
		if pos.x < 0 or pos.x > 800:
			velocities[i].x *= -1
		if pos.y < 0 or pos.y > 600:
			velocities[i].y *= -1
		multimesh.set_instance_transform_2d(i, current_transform)
