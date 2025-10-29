# res://scripts/DemoNoticia.gd
extends Node3D

@export_node_path("AnimationPlayer") var anim_player_path: NodePath
@export var DEFAULT_ANIM := "IDLE"     # Cambia a "INICIO" si tu idle se llama así
@export var BLEND_TIME := 0.35         # Cross-fade entre señas
@export var BLEND_IDLE := 0.25         # Cross-fade hacia/desde idle
@export var IDLE_GAP_SEC := 0.18       # Pausa breve cuando se repite la misma seña

# Lista de palabras para la demo de NOTICIA (en el orden exacto que pediste)
@export var lista_palabras := [
	"INTERESANTE","ULTIMO","AVISAR","INMEDIATO","TELEVISION","CHILE",
	"REGIONES_DE_CHILE","NOVEDAD","NACIONAL","NUEVO","RECIBIR",
	"TELEVISION","SINTOMA","NOTICIA","NOVEDAD","TELEVISION","NOTICIA","INFORMAR"
]

# Mapeo texto -> nombre de animación (sin prefijo de librería)
# Ajusta los valores a los nombres reales tal como aparecen en tu AnimationPlayer.
const MAPEOS := {
	"interesante": "INTERESANTE",
	"ultimo": "ULTIMO",
	"avisar": "AVISAR",
	"inmediato": "INMEDIATO",
	"television": "TELEVISION",
	"chile": "CHILE",
	"regiones_de_chile": "REGIONES_DE_CHILE",
	"novedad": "NOVEDAD",
	"nacional": "NACIONAL",
	"nuevo": "NUEVO",
	"recibir": "RECIBIR",
	"sintoma": "SINTOMA",
	"noticia": "NOTICIA",
	"informar": "INFORMAR"
}

@onready var anim_player: AnimationPlayer = get_node(anim_player_path)

var _run_id: int = 0
var _is_playing: bool = false

func _ready() -> void:
	assert(anim_player, "Asigna el AnimationPlayer en el Inspector.")
	# Reproduce idle al iniciar
	var idle_full := _resolve_anim(DEFAULT_ANIM)
	if idle_full != "":
		anim_player.play(idle_full)
	else:
		push_warning("No se encontró la animación de idle '%s' (en ninguna librería)." % DEFAULT_ANIM)
	# Inicia automáticamente al correr la escena (F6 del editor)
	call_deferred("_start_if_not_playing")

# Reinicia desde el principio con '1' y (opcional) reproduce CONCLUSION con '2'
func _unhandled_input(ev: InputEvent) -> void:
	if ev is InputEventKey and ev.pressed and not ev.echo:
		if ev.keycode == KEY_1:
			reiniciar_desde_el_principio()
			get_viewport().set_input_as_handled()
		elif ev.keycode == KEY_2:
			await reproducir_conclusion()
			get_viewport().set_input_as_handled()

func _start_if_not_playing() -> void:
	if _is_playing:
		return
	_run_id += 1
	_play_demo_sequence_async(_run_id)

func reiniciar_desde_el_principio() -> void:
	_run_id += 1
	_is_playing = false
	if anim_player:
		anim_player.stop()
		var idle_full := _resolve_anim(DEFAULT_ANIM)
		if idle_full != "":
			anim_player.play(idle_full, BLEND_IDLE)
	_play_demo_sequence_async(_run_id)

func _play_demo_sequence_async(run_id: int) -> void:
	_is_playing = true
	var cola: Array[String] = []
	for p in lista_palabras:
		cola.append(_a_nombre_anim(p))
	await _ejecutar_cola(cola, run_id)
	# Si no fue cancelado, vuelve a idle
	if _is_current_run(run_id):
		var idle_full := _resolve_anim(DEFAULT_ANIM)
		if idle_full != "":
			_play(idle_full, BLEND_IDLE)
	_is_playing = false

# (Opcional) tecla 2 → CONCLUSION
func reproducir_conclusion() -> void:
	_run_id += 1
	_is_playing = true
	if anim_player:
		var concl := _resolve_anim("CONCLUSION")
		if concl != "":
			_play(concl, BLEND_TIME)
			await _wait_finish(concl)
		else:
			push_warning("No se encontró la animación 'CONCLUSION' en tus librerías.")
	var idle_full := _resolve_anim(DEFAULT_ANIM)
	if idle_full != "":
		_play(idle_full, BLEND_IDLE)
	_is_playing = false

# ---------- Utilidades ----------

func _a_nombre_anim(palabra: String) -> String:
	var key := palabra.strip_edges().to_lower().replace(" ", "_")
	return MAPEOS.get(key, palabra.strip_edges().to_upper())

# Devuelve el nombre reproducible: "lib/ANIM" si existe, o "" si no existe
func _resolve_anim(nombre_base: String) -> String:
	if anim_player.has_animation(nombre_base):
		return nombre_base
	for lib_key in anim_player.get_animation_library_list():
		var full := "%s/%s" % [lib_key, nombre_base]
		if anim_player.has_animation(full):
			return full
	return ""

func _anim_existe(nombre_base: String) -> bool:
	return _resolve_anim(nombre_base) != ""

func _play(nombre_full: String, blend: float = BLEND_TIME, speed: float = 1.0, from_end := false) -> void:
	anim_player.play(nombre_full, blend, speed, from_end)

func _wait_finish(nombre_full: String) -> void:
	var anim := anim_player.get_animation(nombre_full)
	var length := anim.length if anim else 0.0
	if length <= 0.0:
		length = 0.05
	await get_tree().create_timer(length).timeout

func _sleep(seg: float) -> void:
	if seg > 0.0:
		await get_tree().create_timer(seg).timeout

func _is_same(a: String, b: String) -> bool:
	return a == b

func _is_current_run(run_id: int) -> bool:
	return run_id == _run_id

func _ejecutar_cola(cola: Array[String], run_id: int) -> void:
	var previo := ""
	for nombre_base in cola:
		if not _is_current_run(run_id):
			return
		if not _anim_existe(nombre_base):
			push_warning("No se encontró la animación '%s' en ninguna librería." % nombre_base)
			continue
		var nombre_full := _resolve_anim(nombre_base)

		# Si se repite, mini-gap a idle para evitar que se pegue
		if _is_same(previo, nombre_base):
			var idle_full := _resolve_anim(DEFAULT_ANIM)
			if idle_full != "":
				_play(idle_full, BLEND_IDLE)
				await _sleep(IDLE_GAP_SEC)
				if not _is_current_run(run_id):
					return

		_play(nombre_full, BLEND_TIME)
		await _wait_finish(nombre_full)
		if not _is_current_run(run_id):
			return
		previo = nombre_base
