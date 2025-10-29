extends Node
const C = preload("res://scripts/constants/AnimConfig.gd")
const AnimQueue = preload("res://scripts/Queue/AnimQueue.gd")

signal playing_changed(now_playing: String)
signal queue_changed(queue_text: String)

@export var anim_player_path: NodePath = C.ANIM_PLAYER_PATH
@onready var anim_player: AnimationPlayer = get_node_or_null(anim_player_path)

var q := AnimQueue.new()

var next_timer: Timer
var preblend_timer: Timer
var idle_cycle_timer: Timer

var idle_len: float = 1.0

func _ready() -> void:
	_setup_animplayer()
	_setup_timers()
	_prepare_idle()
	_play_idle_start()
	_emit_info()

func _setup_animplayer():
	if anim_player == null:
		push_error("PlaybackController: AnimationPlayer no encontrado en: " + str(anim_player_path))

func _setup_timers():
	next_timer = Timer.new(); next_timer.one_shot = true; add_child(next_timer)
	preblend_timer = Timer.new(); preblend_timer.one_shot = true; add_child(preblend_timer)
	idle_cycle_timer = Timer.new(); idle_cycle_timer.one_shot = true; add_child(idle_cycle_timer)
	next_timer.timeout.connect(_on_next_timer)
	preblend_timer.timeout.connect(_on_preblend_timeout)
	idle_cycle_timer.timeout.connect(_on_idle_cycle_timeout)
	if anim_player:
		anim_player.animation_finished.connect(_on_anim_finished)

# -------- API pública --------
func append_sequence(txt: String) -> void:
	if anim_player == null or txt == null:
		return
	var cola_estaba_vacia := q.is_empty()
	var to_add := q.append_from_text(txt, func(name): return anim_player.has_animation(name))

	if to_add.is_empty():
		_emit_info()
		return

	if not preblend_timer.is_stopped() and q.size() > 0:
		preblend_timer.stop()

	var cur := (anim_player.current_animation if anim_player else "")
	if (cur == "" or cur == C.DEFAULT_ANIM or cur == C.TMP_IDLE) and cola_estaba_vacia:
		_play_next()
	else:
		_emit_info()

# -------- Idle --------
func _prepare_idle() -> void:
	if anim_player == null: return
	if not anim_player.has_animation(C.DEFAULT_ANIM): return
	var idle: Animation = anim_player.get_animation(C.DEFAULT_ANIM)
	if idle:
		idle.loop_mode = Animation.LOOP_LINEAR
		idle_len = max(0.05, float(idle.length))

func _play_idle_start() -> void:
	if anim_player == null or not anim_player.has_animation(C.DEFAULT_ANIM):
		return
	_stop_all_timers()
	_prepare_idle()
	anim_player.play(C.DEFAULT_ANIM, C.BLEND_IDLE, C.IDLE_SPEED)
	var t: float = max(0.05, idle_len - C.BLEND_IDLE * 0.9)
	idle_cycle_timer.start(t)
	_emit_info()

func _on_idle_cycle_timeout() -> void:
	if anim_player == null: return
	if not anim_player.has_animation(C.DEFAULT_ANIM): return
	var lib: AnimationLibrary = anim_player.get_animation_library("")
	if lib == null: return

	if anim_player.current_animation == C.DEFAULT_ANIM:
		var base: Animation = anim_player.get_animation(C.DEFAULT_ANIM)
		if base:
			if lib.has_animation(C.TMP_IDLE): lib.remove_animation(C.TMP_IDLE)
			var dup: Animation = base.duplicate() as Animation
			dup.loop_mode = Animation.LOOP_LINEAR
			lib.add_animation(C.TMP_IDLE, dup)
			anim_player.play(C.TMP_IDLE, C.BLEND_IDLE, C.IDLE_SPEED)
			anim_player.seek(0.0, true)
	else:
		anim_player.play(C.DEFAULT_ANIM, C.BLEND_IDLE, C.IDLE_SPEED)
		anim_player.seek(0.0, true)

	var t: float = max(0.05, idle_len - C.BLEND_IDLE * 0.9)
	idle_cycle_timer.start(t)

# -------- Reproducción --------
func _current_speed(after_count: int) -> float:
	var spd: float = C.SPEED_BASE
	if after_count >= C.SPEED_THRESH_3:
		spd = C.SPEED_TIER_30
	elif after_count >= C.SPEED_THRESH_2:
		spd = C.SPEED_TIER_20
	elif after_count >= C.SPEED_THRESH_1:
		spd = C.SPEED_TIER_10
	return clamp(spd, 0.1, C.SPEED_MAX)

func _play_next() -> void:
	_stop_all_timers()

	if q.is_empty():
		q.last_played = ""
		_play_idle_start()
		return

	var name: String = q.pop_front()
	var after_count: int = q.size() + 1
	var spd: float = _current_speed(after_count)

	# Idle corto entre repeticiones
	if name == C.GAP_IDLE:
		if anim_player and anim_player.has_animation(C.DEFAULT_ANIM):
			_prepare_idle()
			anim_player.play(C.DEFAULT_ANIM, C.REPEAT_BLEND, C.IDLE_SPEED)
			anim_player.seek(0.0, true)
			var dur_idle: float = clamp(idle_len * C.REPEAT_IDLE_FACTOR, 0.05, C.REPEAT_IDLE_MAX)
			next_timer.start(dur_idle)
			q.last_played = ""
			_emit_info(C.DEFAULT_ANIM)
			return

	var base_anim: Animation = (anim_player.get_animation(name) if anim_player else null)
	if base_anim:
		base_anim.loop_mode = Animation.LOOP_NONE

	# Duplicado temporal para repetir misma seña
	if name == q.last_played and base_anim:
		var lib: AnimationLibrary = anim_player.get_animation_library("")
		if lib:
			if lib.has_animation(C.TMP_REPEAT): lib.remove_animation(C.TMP_REPEAT)
			var dup: Animation = base_anim.duplicate() as Animation
			if dup:
				dup.loop_mode = Animation.LOOP_NONE
				lib.add_animation(C.TMP_REPEAT, dup)
				anim_player.play(C.TMP_REPEAT, C.BLEND_TIME, spd)
				anim_player.seek(0.0, true)
			else:
				anim_player.play(name, 0.0, spd)
				anim_player.seek(0.0, true)
	else:
		anim_player.play(name, C.BLEND_TIME, spd)

	var dur: float = 0.5
	if base_anim:
		dur = max(0.05, float(base_anim.length))
	dur = dur / spd
	next_timer.start(dur)

	# Preblend a idle si no llegan más entradas
	if q.is_empty() and anim_player.has_animation(C.DEFAULT_ANIM):
		var pre_time: float = max(0.0, dur - C.BLEND_IDLE * 0.9)
		if pre_time > 0.0:
			preblend_timer.start(pre_time)
		else:
			_on_preblend_timeout()

	q.last_played = name
	_emit_info(name)

# -------- Timers / señales --------
func _on_preblend_timeout() -> void:
	_play_idle_start()

func _on_next_timer() -> void:
	if q.is_empty():
		_play_idle_start()
	else:
		_play_next()

func _on_anim_finished(anim_name: String) -> void:
	if not next_timer.is_stopped():
		next_timer.stop()
	if not preblend_timer.is_stopped():
		preblend_timer.stop()
	if anim_name == C.DEFAULT_ANIM or anim_name == C.TMP_IDLE:
		return
	if q.is_empty():
		_play_idle_start()
	else:
		_play_next()

func _stop_all_timers():
	if next_timer and not next_timer.is_stopped():
		next_timer.stop()
	if preblend_timer and not preblend_timer.is_stopped():
		preblend_timer.stop()
	if idle_cycle_timer and not idle_cycle_timer.is_stopped():
		idle_cycle_timer.stop()

# -------- UI helpers --------
func _emit_info(now_playing: String = ""):
	playing_changed.emit(now_playing if now_playing != "" else "Idle")
	queue_changed.emit(q.as_text())
