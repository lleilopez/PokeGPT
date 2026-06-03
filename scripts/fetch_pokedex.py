"""
fetch_pokedex.py — Descarga datos de la Pokédex desde PokeAPI y los guarda
como texto plano en data/raw/pokedex.txt.

PokeAPI es una API pública y gratuita (pokeapi.co). No requiere clave.
No es una API de IA: es simplemente una fuente de datos estructurados.

Por defecto descarga los 151 Pokémon de la primera generación.
El rango se puede cambiar con las variables POKE_INICIO y POKE_FIN.

Uso:
    python scripts/fetch_pokedex.py

Salida:
    data/raw/pokedex.txt — una entrada por Pokémon, formato prosa en español
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── Configuración ────────────────────────────────────────────────────────────

BASE_URL    = "https://pokeapi.co/api/v2"
DATA_DIR    = os.getenv("DATA_DIR", "data")
SALIDA      = os.path.join(DATA_DIR, "raw", "pokedex.txt")
POKE_INICIO = 1     # ID del primer Pokémon a descargar
POKE_FIN    = 1025  # ID del último Pokémon a descargar (Gen 1-9 completas)
PAUSA       = 0.2   # Segundos entre peticiones para no saturar la API

# ─── Traducciones hardcodeadas (18 tipos, sin peticiones extra) ───────────────

TIPOS_ES = {
    "normal":   "Normal",
    "fire":     "Fuego",
    "water":    "Agua",
    "electric": "Electrico",
    "grass":    "Planta",
    "ice":      "Hielo",
    "fighting": "Lucha",
    "poison":   "Veneno",
    "ground":   "Tierra",
    "flying":   "Volador",
    "psychic":  "Psiquico",
    "bug":      "Bicho",
    "rock":     "Roca",
    "ghost":    "Fantasma",
    "dragon":   "Dragon",
    "dark":     "Siniestro",
    "steel":    "Acero",
    "fairy":    "Hada",
}

# ─── Caché en memoria para evitar peticiones repetidas ────────────────────────
# Muchos Pokémon comparten habilidades y movimientos, así que los cacheamos.

_cache_habilidades: dict[str, str] = {}
_cache_movimientos: dict[str, str] = {}


# ─── Funciones de acceso a la API ─────────────────────────────────────────────

def get_json(url: str) -> dict | None:
    """Realiza una petición GET y devuelve el JSON. Devuelve None si hay error."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"    [ERROR] {url} -> {e}")
        return None


def nombre_es(lista_nombres: list, fallback: str) -> str:
    """Extrae el nombre en español de una lista de objetos {name, language}."""
    for entrada in lista_nombres:
        if entrada["language"]["name"] == "es":
            return entrada["name"]
    return fallback


def flavor_text_es(lista_textos: list) -> str:
    """
    Extrae el texto de la Pokédex en español.
    Limpia saltos de línea y caracteres de control que vienen de la API.
    """
    for entrada in lista_textos:
        if entrada["language"]["name"] == "es":
            texto = entrada["flavor_text"]
            # La API devuelve saltos de línea y avances de forma (\f, \n)
            texto = texto.replace("\n", " ").replace("\f", " ")
            # Colapsar espacios dobles
            while "  " in texto:
                texto = texto.replace("  ", " ")
            return texto.strip()
    return ""


def habilidad_es(nombre_api: str) -> str:
    """Devuelve el nombre de una habilidad en español (con caché)."""
    if nombre_api in _cache_habilidades:
        return _cache_habilidades[nombre_api]

    data = get_json(f"{BASE_URL}/ability/{nombre_api}")
    time.sleep(PAUSA)

    if data:
        nombre = nombre_es(data.get("names", []), nombre_api)
    else:
        nombre = nombre_api

    _cache_habilidades[nombre_api] = nombre
    return nombre


def movimiento_es(nombre_api: str) -> str:
    """Devuelve el nombre de un movimiento en español (con caché)."""
    if nombre_api in _cache_movimientos:
        return _cache_movimientos[nombre_api]

    data = get_json(f"{BASE_URL}/move/{nombre_api}")
    time.sleep(PAUSA)

    if data:
        nombre = nombre_es(data.get("names", []), nombre_api)
    else:
        nombre = nombre_api

    _cache_movimientos[nombre_api] = nombre
    return nombre


# ─── Construcción de la entrada de texto ──────────────────────────────────────

def construir_entrada(poke_id: int) -> str | None:
    """
    Descarga toda la información de un Pokémon y la convierte a prosa en español.
    Devuelve None si no se pudo obtener la información básica.
    """

    # --- Datos principales: stats, tipos, habilidades, movimientos ---
    poke = get_json(f"{BASE_URL}/pokemon/{poke_id}")
    time.sleep(PAUSA)
    if not poke:
        return None

    # --- Datos de especie: nombre ES, texto Pokédex ES ---
    especie = get_json(f"{BASE_URL}/pokemon-species/{poke_id}")
    time.sleep(PAUSA)
    if not especie:
        return None

    # Nombre en español
    nombre = nombre_es(especie.get("names", []), poke["name"].capitalize())

    # Tipos en español
    tipos_raw = [t["type"]["name"] for t in poke["types"]]
    tipos_es  = [TIPOS_ES.get(t, t.capitalize()) for t in tipos_raw]
    tipos_str = " y ".join(tipos_es)

    # Stats base
    stats = {s["stat"]["name"]: s["base_stat"] for s in poke["stats"]}
    hp    = stats.get("hp", "?")
    atk   = stats.get("attack", "?")
    defe  = stats.get("defense", "?")
    spatk = stats.get("special-attack", "?")
    spdef = stats.get("special-defense", "?")
    vel   = stats.get("speed", "?")

    # Habilidades en español (excluimos las ocultas para simplificar)
    habilidades_raw = [
        h["ability"]["name"]
        for h in poke["abilities"]
        if not h["is_hidden"]
    ]
    habilidades_es = [habilidad_es(h) for h in habilidades_raw]
    habilidades_str = " o ".join(habilidades_es) if habilidades_es else "Desconocida"

    # Movimientos aprendidos por subida de nivel (máximo 6, orden por nivel)
    movs_nivel = [
        m for m in poke["moves"]
        if any(d["move_learn_method"]["name"] == "level-up"
               for d in m["version_group_details"])
    ]
    # Ordenamos por el nivel en que se aprende (primera versión disponible)
    def nivel_aprendizaje(m):
        detalles = [
            d["level_learned_at"]
            for d in m["version_group_details"]
            if d["move_learn_method"]["name"] == "level-up"
        ]
        return min(detalles) if detalles else 999

    movs_nivel.sort(key=nivel_aprendizaje)
    nombres_movs = [movimiento_es(m["move"]["name"]) for m in movs_nivel[:6]]
    movs_str = ", ".join(nombres_movs) if nombres_movs else "Ninguno"

    # Texto Pokédex en español
    flavor = flavor_text_es(especie.get("flavor_text_entries", []))

    # --- Construcción de la prosa ---
    partes = [
        f"{nombre} es un Pokemon de tipo {tipos_str}.",
    ]
    if flavor:
        partes.append(flavor)
    partes.append(
        f"Tiene {hp} de HP, {atk} de Ataque, {defe} de Defensa, "
        f"{spatk} de Ataque Especial, {spdef} de Defensa Especial "
        f"y {vel} de Velocidad."
    )
    partes.append(f"Sus habilidades son {habilidades_str}.")
    partes.append(f"Puede aprender movimientos como {movs_str}.")

    return " ".join(partes)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Descarga de Pokédex desde PokeAPI")
    print(f"  Pokemon #{POKE_INICIO} — #{POKE_FIN}")
    print(f"  Tiempo estimado: 25-40 min (con caché de habilidades/movimientos)")
    print("=" * 55)

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)

    entradas = []
    total    = POKE_FIN - POKE_INICIO + 1
    errores  = 0

    for poke_id in range(POKE_INICIO, POKE_FIN + 1):
        print(f"  [{poke_id:>3}/{POKE_FIN}] Descargando...", end=" ", flush=True)
        entrada = construir_entrada(poke_id)
        if entrada:
            entradas.append(entrada)
            print("OK")
        else:
            errores += 1
            print("ERROR (se omite)")

    # Guardamos una entrada por línea, separadas por línea en blanco
    with open(SALIDA, "w", encoding="utf-8") as f:
        f.write("\n\n".join(entradas) + "\n")

    print()
    print(f"Guardado en: {SALIDA}")
    print(f"Entradas escritas : {len(entradas)} / {total}")
    print(f"Errores           : {errores}")
    print(f"Caracteres totales: {sum(len(e) for e in entradas):,}")
    print("=" * 55)


if __name__ == "__main__":
    main()
