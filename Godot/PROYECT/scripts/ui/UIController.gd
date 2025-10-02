extends Node
signal submit_text(txt: String)

const C = preload("res://scripts/constants/AnimConfig.gd")

@export var line_path: NodePath  = C.LINEEDIT_PATH
@export var button_path: NodePath = C.BUTTON_PATH
@export var label_path: NodePath  = C.LABEL_PATH

@onready var line: LineEdit = get_node_or_null(line_path)
@onready var button: Button = get_node_or_null(button_path)
@onready var label: Label = get_node_or_null(label_path)

func _ready() -> void:
	if button:
		button.pressed.connect(_on_button)
	if line:
		line.text_submitted.connect(_on_submit)

func _on_button() -> void:
	if line:
		submit_text.emit(line.text)
		# line.text = ""

func _on_submit(txt: String) -> void:
	submit_text.emit(txt)
	# line.text = ""

func update_info(now_playing: String, queue_text: String) -> void:
	if label:
		label.text = "%s | Cola: %s" % [now_playing, queue_text]
