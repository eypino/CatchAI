extends RefCounted
const C = preload("res://scripts/constants/AnimConfig.gd")

var queue: Array[String] = []
var last_played: String = ""   # la actualiza el reproductor

func is_empty() -> bool:
	return queue.is_empty()

func size() -> int:
	return queue.size()

# Renombrado para no chocar con Object.to_string()
func as_text() -> String:
	return ", ".join(queue) if queue.size() > 0 else "(vacía)"

# Último nombre real en cola (ignora GAP_IDLE); si no hay, usa last_played
func effective_tail_name() -> String:
	for i in range(queue.size() - 1, -1, -1):
		var n: String = queue[i]
		if n != C.GAP_IDLE:
			return n
	return last_played

func _parse_valid_names(txt: String, has_anim: Callable) -> Array[String]:
	var parts: Array[String] = []
	for p in txt.split(","):
		var name := (p as String).strip_edges()
		if name != "" and has_anim.call(name):
			parts.append(name)
	return parts

func _insert_idle_on_repeats(seq: Array[String]) -> Array[String]:
	if seq.is_empty():
		return seq
	var out: Array[String] = []
	var prev := ""
	for name in seq:
		if prev != "" and name == prev:
			out.append(C.GAP_IDLE)
		out.append(name)
		prev = name
	return out

# Devuelve lo realmente añadido (incluye GAPs)
func append_from_text(txt: String, has_anim: Callable) -> Array[String]:
	var names := _parse_valid_names(txt, has_anim)
	if names.is_empty():
		return []
	var to_add := _insert_idle_on_repeats(names)

	# Si el primer nuevo coincide con el último efectivo actual, inserta GAP al inicio
	var tail := effective_tail_name()
	if tail != "" and to_add.size() > 0 and to_add[0] == tail:
		to_add.insert(0, C.GAP_IDLE)

	for n in to_add:
		queue.append(n)
	return to_add

func pop_front() -> String:
	return queue.pop_front()

func clear():
	queue.clear()
