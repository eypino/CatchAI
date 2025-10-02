# GlossRouter.gd
extends RefCounted
class_name GlossRouter

## Convenciones de nombres:
## - Palabras completas en MAYÚSCULAS (sin tildes)
## - Ñ como "NN"
## - Letras: "A".."Z" y "NN"
## Ajusta si tu AnimationPlayer usa otra convención.

static func _strip_accents(s: String) -> String:
	var map := {
		"Á":"A","É":"E","Í":"I","Ó":"O","Ú":"U","Ü":"U",
		"á":"A","é":"E","í":"I","ó":"O","ú":"U","ü":"U"
	}
	var out := ""
	for ch in s:
		out += map.get(ch, ch)
	return out

static func normalize_glosa(g: String) -> String:
	if g.is_empty():
		return g
	var up := _strip_accents(g).to_upper()
	# Convención: Ñ -> NN
	up = up.replace("Ñ", "NN")
	# Si tu AnimationPlayer usa espacios tal cual, NO toques los espacios.
	# Si en tu proyecto usas guiones bajos, descomenta la línea:
	# up = up.replace(" ", "_")
	return up

static func is_single_letter(glosa_norm: String) -> bool:
	# Considera "A".. "Z" o "NN" como letras válidas para deletreo
	if glosa_norm == "NN":
		return true
	return glosa_norm.length() == 1 and glosa_norm[0] >= 'A' and glosa_norm[0] <= 'Z'

static func spell_word(word_norm: String) -> PackedStringArray:
	# Deletreo por letras (Ñ ya viene normalizada como "NN")
	var out := PackedStringArray()
	var i := 0
	while i < word_norm.length():
		var ch := word_norm[i]
		if ch == 'N' and i + 1 < word_norm.length() and word_norm[i + 1] == 'N':
			out.append("NN")
			i += 2
		else:
			out.append(String(ch))
			i += 1
	return out

static func route_glosas_to_anims(glosas: Array, anim_set: Dictionary) -> PackedStringArray:
	# anim_set: { "ANIM_NAME": true, ... } para O(1) existencia
	var result := PackedStringArray()

	for g in glosas:
		if typeof(g) != TYPE_STRING:
			continue
		var norm := normalize_glosa(g)

		# 1) Existe anim con el nombre exacto
		if anim_set.has(norm):
			result.append(norm)
			continue

		# 2) Si no existe y parece palabra (no letra), fallback a deletreo
		if not is_single_letter(norm):
			var spelled := spell_word(norm)
			for letter in spelled:
				if anim_set.has(letter):
					result.append(letter)
			continue

		# 3) Si es letra sola, intenta directo
		if anim_set.has(norm):
			result.append(norm)
		# Si no existe ni la letra, se omite (o pon aquí un "DESCONOCIDO")
		# else:
		#     result.append("DESCONOCIDO")

	return result
