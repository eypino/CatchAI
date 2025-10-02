extends Node
class_name WsClient

# ---- Config ----
@export var ws_url: String = "ws://localhost:8000/ws"
@export var fixed_latency_sec: float = 0.0

# Arrastra aquí tu AnimationPlayer (desde el Inspector).
# Con tu árbol actual suele ser "../../AnimationPlayer".
@export var anim_player_path: NodePath = NodePath("../../AnimationPlayer")

# (Opcional) endpoint demo para empujar datos vía HTTP y que el server los envíe por WS
@export var demo_http_url: String = "http://localhost:8000/demo"

# ---- Deps ----
const GlossRouter := preload("res://scripts/sign/GlossRouter.gd")

# ---- Señales públicas ----
signal ws_connected
signal ws_disconnected
signal ws_error(err_code: int)
signal anim_list_ready(anims: PackedStringArray)

# ---- Estado interno ----
var _ws: WebSocketPeer = WebSocketPeer.new()
var _anim_player: AnimationPlayer
var _anim_set: Dictionary = {}     # { "ANIM_NAME": true }
var _did_emit_connected: bool = false
var _reconnect_delay: float = 1.0

func _ready() -> void:
	# Resolver AnimationPlayer (por path exportado).
	_anim_player = get_node_or_null(anim_player_path)
	if _anim_player == null:
		push_warning("[WS] AnimationPlayer no encontrado aún. Ajusta 'anim_player_path'. Conecto WS igual.")
	else:
		_rebuild_anim_index()

	_connect_ws()
	set_process(true)

func _process(_delta: float) -> void:
	# Intento perezoso: si no había anim_player, vuelve a buscarlo
	if _anim_player == null:
		_anim_player = get_node_or_null(anim_player_path)
		if _anim_player != null:
			_rebuild_anim_index()

	var state: int = _ws.get_ready_state()

	# Procesar IO
	if state == WebSocketPeer.STATE_CONNECTING or state == WebSocketPeer.STATE_OPEN:
		_ws.poll()

	if state == WebSocketPeer.STATE_OPEN:
		if not _did_emit_connected:
			_did_emit_connected = true
			print("[WS] Conectado a %s" % ws_url)
			emit_signal("ws_connected")
		_read_packets()
	elif state == WebSocketPeer.STATE_CLOSED:
		if _did_emit_connected:
			_did_emit_connected = false
			print("[WS] Desconectado")
			emit_signal("ws_disconnected")
		await get_tree().create_timer(_reconnect_delay).timeout
		_connect_ws()

func _exit_tree() -> void:
	if _ws.get_ready_state() == WebSocketPeer.STATE_OPEN:
		_ws.close()

# ---- Conexión ----
func _connect_ws() -> void:
	var err: int = _ws.connect_to_url(ws_url)
	if err != OK:
		print("[WS] Error al conectar: %s" % str(err))
		emit_signal("ws_error", err)
	else:
		print("[WS] Conectando a %s..." % ws_url)

# ---- Recepción ----
func _read_packets() -> void:
	while _ws.get_available_packet_count() > 0:
		var bytes: PackedByteArray = _ws.get_packet()
		var text: String = bytes.get_string_from_utf8()
		_process_text_message(text)

func _process_text_message(text: String) -> void:
	var parsed: Variant = JSON.parse_string(text)
	if parsed == null:
		print("[WS] JSON inválido (descartado)")
		return

	var segments: Array = []
	if typeof(parsed) == TYPE_ARRAY:
		segments = parsed
	elif typeof(parsed) == TYPE_DICTIONARY:
		segments = [parsed]
	else:
		return

	for seg in segments:
		if typeof(seg) != TYPE_DICTIONARY:
			continue
		var seg_dict: Dictionary = seg

		# Obtener glosas tolerando string tipo "['A','B']"
		var glosas: Array = []
		var raw_glosas: Variant = seg_dict.get("glosas", [])
		if typeof(raw_glosas) == TYPE_ARRAY:
			glosas = raw_glosas
		elif typeof(raw_glosas) == TYPE_STRING:
			var s: String = raw_glosas
			var parsed_g: Variant = JSON.parse_string(s.replace("'", "\""))
			if typeof(parsed_g) == TYPE_ARRAY:
				glosas = parsed_g

		if glosas.is_empty():
			continue

		# Mapear glosas a animaciones (con fallback a deletreo)
		if _anim_player == null:
			continue
		var anims: PackedStringArray = GlossRouter.route_glosas_to_anims(glosas, _anim_set)
		if anims.is_empty():
			continue

		if fixed_latency_sec > 0.0:
			await get_tree().create_timer(fixed_latency_sec).timeout

		emit_signal("anim_list_ready", anims)

# ---- Utilidades ----
func _rebuild_anim_index() -> void:
	_anim_set.clear()
	if _anim_player == null:
		return
	var names: PackedStringArray = _anim_player.get_animation_list()
	for name in names:
		var name_str: String = name
		_anim_set[GlossRouter.normalize_glosa(name_str)] = true

# Estado textual para el botón de prueba
func get_status_text() -> String:
	var s: int = _ws.get_ready_state()
	match s:
		WebSocketPeer.STATE_CONNECTING: return "CONNECTING"
		WebSocketPeer.STATE_OPEN:       return "OPEN"
		WebSocketPeer.STATE_CLOSING:    return "CLOSING"
		WebSocketPeer.STATE_CLOSED:     return "CLOSED"
		_:                              return "UNKNOWN"

# Disparar /demo por HTTP (si tu server lo tiene)
func trigger_demo() -> void:
	var http: HTTPRequest = HTTPRequest.new()
	add_child(http)
	http.request_completed.connect(func(_res: int, code: int, _h: PackedStringArray, _b: PackedByteArray):
		print("[WS-TEST] Demo solicitada -> code=", code)
		http.queue_free()
	)
	var err: int = http.request(demo_http_url)
	if err != OK:
		print("[WS-TEST] Error al solicitar demo: ", err)
		http.queue_free()
