"""
explore_dataset.py — Carga el texto crudo de la Pokédex y muestra estadísticas clave.

Este script es el primer contacto con los datos antes de tokenizar. El objetivo
es entender qué tenemos: cuánto texto, qué caracteres aparecen, con qué frecuencia,
y cuántas secuencias de entrenamiento podríamos generar.

Uso:
    python src/explore_dataset.py
"""

import os
from collections import Counter
from dotenv import load_dotenv

# Cargamos las variables de entorno definidas en .env
load_dotenv()
DATA_DIR     = os.getenv("DATA_DIR", "data")
CONTEXT_LEN  = int(os.getenv("CONTEXT_LENGTH", 128))

# Ruta al texto crudo
RUTA_DATASET = os.path.join(DATA_DIR, "raw", "pokedex.txt")


def cargar_texto(ruta: str) -> str:
    """Lee el archivo de texto crudo y devuelve su contenido completo."""
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read()


def stats_basicas(texto: str) -> dict:
    """
    Calcula métricas básicas del corpus:
    - Total de caracteres
    - Total de líneas (entradas de Pokémon)
    - Total de palabras (estimación)
    - Vocabulario de caracteres únicos
    """
    lineas      = [l for l in texto.splitlines() if l.strip()]
    palabras    = texto.split()
    vocab       = sorted(set(texto))

    return {
        "total_chars":   len(texto),
        "total_lineas":  len(lineas),
        "total_palabras": len(palabras),
        "vocab_size":    len(vocab),
        "vocab":         vocab,
    }


def frecuencia_chars(texto: str, top_n: int = 20) -> list[tuple[str, int]]:
    """Devuelve los top_n caracteres más frecuentes del corpus."""
    contador = Counter(texto)
    return contador.most_common(top_n)


def secuencias_posibles(total_chars: int, context_len: int) -> int:
    """
    Estima cuántas secuencias de entrenamiento podemos extraer.
    En entrenamiento autoregresivo desplazamos la ventana de 1 en 1:
    cada posición genera una secuencia de entrada y su correspondiente target.
    """
    return max(0, total_chars - context_len)


def imprimir_muestra(texto: str, inicio: int = 0, fin: int = 300):
    """Imprime un fragmento del texto para inspeccionarlo visualmente."""
    print(texto[inicio:fin])


def main():
    print("=" * 55)
    print("  Exploración del dataset — PokeGPT V0.1")
    print("=" * 55)

    # 1. Carga del texto
    print(f"\nArchivo: {RUTA_DATASET}")
    texto = cargar_texto(RUTA_DATASET)
    print("Texto cargado correctamente.\n")

    # 2. Estadísticas básicas
    stats = stats_basicas(texto)
    print("--- Estadísticas básicas ---")
    print(f"  Caracteres totales : {stats['total_chars']:,}")
    print(f"  Líneas (Pokémon)   : {stats['total_lineas']:,}")
    print(f"  Palabras (aprox.)  : {stats['total_palabras']:,}")
    print(f"  Vocabulario        : {stats['vocab_size']} caracteres únicos")

    # 3. Vocabulario completo
    print("\n--- Vocabulario de caracteres ---")
    vocab_repr = [repr(c) for c in stats["vocab"]]
    print("  " + "  ".join(vocab_repr))

    # 4. Caracteres más frecuentes
    print("\n--- Top 20 caracteres más frecuentes ---")
    frecuencias = frecuencia_chars(texto, top_n=20)
    for char, count in frecuencias:
        barra = "#" * (count // 100)   # barra visual proporcional
        print(f"  {repr(char):8s}  {count:6,}  {barra}")

    # 5. Secuencias de entrenamiento estimadas
    n_seq = secuencias_posibles(stats["total_chars"], CONTEXT_LEN)
    print(f"\n--- Secuencias de entrenamiento (context_len={CONTEXT_LEN}) ---")
    print(f"  Secuencias posibles: {n_seq:,}")
    print(f"  (Una por cada posición de la ventana deslizante)")

    # 6. Muestra del texto
    print("\n--- Primeros 400 caracteres del corpus ---")
    print("-" * 45)
    imprimir_muestra(texto, 0, 400)
    print("-" * 45)

    print("\n[OK] Exploración completada.")
    print("=" * 55)


if __name__ == "__main__":
    main()
