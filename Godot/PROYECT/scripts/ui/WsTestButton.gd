extends Button

# Paths relativos a TU árbol (desde BtnTestWS hasta WsClient y LblWsStatus)
@export var ws_client_path: NodePath = "../../../../Controllers/WsClient"
@export var status_label_path: NodePath = "../LblWsStatus"

@onready var ws: Node = get_node_or_null(ws_client_path)
@onready var status_label: Label = get_node_or_null(status_label_path)

func _ready() -> void:
	pressed.connect(_on_pressed)
	_update_status()
	# refresca estado cada 1 s
	var t := Timer.new()
	t.wait_time = 1.0
	t.autostart = true
	t.timeout.connect(_update_status)
	add_child(t)

func _on_pressed() -> void:
	if ws and ws.has_method("get_status_text"):
		var s: String = ws.get_status_text()
		print("[WS-TEST] Estado actual: ", s)
		# Si dejaste /demo en FastAPI
		if ws.has_method("trigger_demo"):
			ws.trigger_demo()
	else:
		print("[WS-TEST] No encuentro WsClient en: ", ws_client_path)

func _update_status() -> void:
	if not status_label:
		return
	if ws and ws.has_method("get_status_text"):
		status_label.text = "WS: %s" % ws.get_status_text()
	else:
		status_label.text = "WS: (sin cliente)"
