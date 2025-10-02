extends Node3D

@export var playback_controller_path: NodePath = "Controllers/PlaybackController"
@export var ui_controller_path: NodePath       = "Controllers/UIController"

@onready var playback: Node = get_node_or_null(playback_controller_path)
@onready var ui: Node       = get_node_or_null(ui_controller_path)
@onready var _ws: Node      = $"Controllers/WsClient"   # WS → llega lista de anims

func _ready() -> void:
	# --- UI ---
	if ui:
		if ui.has_signal("submit_text"):
			ui.submit_text.connect(_on_submit_text)

	# --- Playback (para refrescar info en pantalla) ---
	if playback:
		if playback.has_signal("playing_changed"):
			playback.playing_changed.connect(func(_now): _refresh_info())
		if playback.has_signal("queue_changed"):
			playback.queue_changed.connect(func(_q): _refresh_info())

	# --- WebSocket → Cola ---
	if _ws:
		if _ws.has_signal("anim_list_ready"):
			_ws.anim_list_ready.connect(_on_ws_anim_list_ready)
		if _ws.has_signal("ws_connected"):
			_ws.ws_connected.connect(func(): print("[RUNNER] WS conectado"))
		if _ws.has_signal("ws_disconnected"):
			_ws.ws_disconnected.connect(func(): print("[RUNNER] WS desconectado"))
		if _ws.has_signal("ws_error"):
			_ws.ws_error.connect(func(err): print("[RUNNER] WS error: %s" % str(err)))

	_refresh_info()


func _on_submit_text(txt: String) -> void:
	if playback and playback.has_method("append_sequence"):
		playback.append_sequence(txt)


func _refresh_info() -> void:
	if not ui or not playback:
		return
	var head: String = ""
	if "anim_player" in playback and playback.anim_player:
		head = str(playback.anim_player.current_animation)
	if head == "" or head == "null":
		head = "Idle"
	if "q" in playback and ui.has_method("update_info"):
		ui.update_info(head, playback.q.as_text())


# -------- WS → Encolado --------
func _on_ws_anim_list_ready(anims: PackedStringArray) -> void:
	_enqueue_many_best_effort(anims)

func _enqueue_many_best_effort(anims: PackedStringArray) -> void:
	# 1) Si tienes un nodo AnimQueue con enqueue_many()
	var anim_queue: Node = get_node_or_null("Controllers/AnimQueue")
	if anim_queue and anim_queue.has_method("enqueue_many"):
		anim_queue.enqueue_many(anims)
		return

	# 2) Si tu PlaybackController expone enqueue_many()
	if playback and playback.has_method("enqueue_many"):
		playback.enqueue_many(anims)
		return

	# 3) Fallback: usa append_sequence() con coma para reutilizar tu parser existente
	if playback and playback.has_method("append_sequence"):
		var joined: String = ",".join(anims)
		playback.append_sequence(joined)
		return

	push_error("[RUNNER] No encontré cómo encolar animaciones (ni AnimQueue.enqueue_many, ni Playback.enqueue_many, ni append_sequence). Ajusta rutas o expón uno de esos métodos.")
