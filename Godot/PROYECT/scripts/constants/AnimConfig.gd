extends RefCounted

# RUTAS según tu escena
const ANIM_PLAYER_PATH := "Animaciones_Total_03/recording_16_47_21_gmt-3_rig/AnimationPlayer"
const LINEEDIT_PATH    := "CanvasLayer/Control/HBoxContainer/LineEdit"
const BUTTON_PATH      := "CanvasLayer/Control/Button"
const LABEL_PATH       := "CanvasLayer/Control/HBoxContainer/Label"

# NOMBRES ESPECIALES
const DEFAULT_ANIM := "INICIO"
const TMP_REPEAT   := "_repeat_tmp"
const TMP_IDLE     := "_idle_tmp"
const GAP_IDLE     := "__IDLE_GAP__"

# BLENDS
const BLEND_TIME   := 0.35
const BLEND_IDLE   := 0.35
const REPEAT_BLEND := 0.15

# Idle intermedio entre repeticiones
const REPEAT_IDLE_FACTOR := 0.25
const REPEAT_IDLE_MAX    := 0.60
const IDLE_SPEED := 1.0

# Tiers de velocidad
const SPEED_THRESH_1 := 10
const SPEED_THRESH_2 := 20
const SPEED_THRESH_3 := 30
const SPEED_BASE     := 1.00
const SPEED_TIER_10  := 1.15
const SPEED_TIER_20  := 1.35
const SPEED_TIER_30  := 1.60
const SPEED_MAX      := 2.00
