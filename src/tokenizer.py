"""
tokenizer.py — Tokenizador por caracteres para PokeGPT.

Un tokenizador convierte texto (cadena de caracteres) en una secuencia de
números enteros que la red neuronal puede procesar, y viceversa.

¿Por qué caracteres y no palabras o subpalabras (BPE)?
- Es el enfoque más simple posible: cada carácter único es un token.
- El vocabulario es pequeño (~88 tokens frente a los ~50.000 de GPT-2).
- No necesita ninguna librería externa: se construye directamente del corpus.
- Ideal para entender los fundamentos antes de escalar a tokenizadores más complejos.

Flujo completo:
    texto crudo  →  encode()  →  [3, 45, 12, 7, ...]  →  modelo neuronal
    modelo       →  decode()  →  texto generado

Archivos relacionados:
    data/raw/pokedex.txt         — corpus de entrenamiento
    data/processed/vocab.json    — vocabulario guardado en disco
"""

import json
import os


class CharTokenizer:
    """
    Tokenizador a nivel de carácter.

    Atributos principales:
        char2idx  — dict que mapea cada carácter a su índice entero
        idx2char  — dict inverso: índice entero → carácter
        vocab_size — número total de tokens (caracteres únicos)
    """

    # Token especial para caracteres no vistos durante el entrenamiento.
    # Lo añadimos al vocabulario con índice 0 para que siempre exista.
    TOKEN_UNK = "<UNK>"

    def __init__(self):
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self.vocab_size: int = 0

    # ─── Construcción del vocabulario ─────────────────────────────────────────

    def build_vocab(self, texto: str) -> None:
        """
        Construye el vocabulario a partir de un corpus de texto.

        El proceso es simple:
          1. Extraer todos los caracteres únicos del corpus.
          2. Ordenarlos para que el vocabulario sea determinista
             (mismo corpus → mismo vocab en cualquier ejecución).
          3. Asignar a cada carácter un índice entero empezando en 1.
             El índice 0 se reserva para el token especial <UNK>.

        Args:
            texto: el corpus completo como una sola cadena de texto.
        """
        # Reservamos el índice 0 para el token desconocido
        self.char2idx = {self.TOKEN_UNK: 0}
        self.idx2char = {0: self.TOKEN_UNK}

        # sorted() garantiza orden alfabético/Unicode determinista
        chars_unicos = sorted(set(texto))

        for i, char in enumerate(chars_unicos, start=1):
            self.char2idx[char] = i
            self.idx2char[i]    = char

        # vocab_size incluye el token <UNK>
        self.vocab_size = len(self.char2idx)

    # ─── Encode / Decode ──────────────────────────────────────────────────────

    def encode(self, texto: str) -> list[int]:
        """
        Convierte una cadena de texto en una lista de índices enteros.

        Los caracteres no presentes en el vocabulario se mapean a <UNK> (índice 0).
        Esto hace que el tokenizador sea robusto ante texto nuevo en inferencia.

        Ejemplo:
            encode("Abc") → [34, 2, 15]  (índices dependen del vocab construido)

        Args:
            texto: cadena a tokenizar.

        Returns:
            Lista de enteros, uno por carácter.
        """
        return [self.char2idx.get(ch, 0) for ch in texto]

    def decode(self, indices: list[int]) -> str:
        """
        Convierte una lista de índices enteros de vuelta a texto.

        Los índices desconocidos (fuera del vocab) se reemplazan por '?'.

        Ejemplo:
            decode([34, 2, 15]) → "Abc"

        Args:
            indices: lista de enteros producida por encode() o por el modelo.

        Returns:
            Cadena de texto reconstruida.
        """
        return "".join(self.idx2char.get(i, "?") for i in indices)

    # ─── Persistencia en disco ────────────────────────────────────────────────

    def save(self, ruta: str) -> None:
        """
        Guarda el vocabulario en un archivo JSON legible por humanos.

        Guardamos char2idx porque es suficiente para reconstruir todo:
        idx2char es simplemente el diccionario inverso.

        Args:
            ruta: ruta donde guardar el archivo (ej. data/processed/vocab.json).
        """
        os.makedirs(os.path.dirname(ruta), exist_ok=True)

        datos = {
            "vocab_size": self.vocab_size,
            "char2idx":   self.char2idx,
        }

        with open(ruta, "w", encoding="utf-8") as f:
            # ensure_ascii=False para que los caracteres especiales (ñ, á...)
            # se guarden legibles en lugar de secuencias \uXXXX
            json.dump(datos, f, ensure_ascii=False, indent=2)

    def load(self, ruta: str) -> None:
        """
        Carga un vocabulario previamente guardado con save().

        Args:
            ruta: ruta al archivo JSON del vocabulario.
        """
        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)

        self.char2idx  = datos["char2idx"]
        self.vocab_size = datos["vocab_size"]

        # Las claves de JSON son siempre strings, pero idx2char necesita int
        self.idx2char = {int(i): ch for ch, i in self.char2idx.items()}

    # ─── Utilidades ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


# ─── Script de prueba ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Ejecuta una demostración completa del tokenizador:
      1. Carga el corpus
      2. Construye el vocabulario
      3. Guarda el vocab en disco
      4. Verifica encode → decode (debe reproducir el texto original)
      5. Imprime ejemplos y estadísticas
    """
    from dotenv import load_dotenv
    load_dotenv()

    DATA_DIR      = os.getenv("DATA_DIR", "data")
    CORPUS_PATH   = os.path.join(DATA_DIR, "raw", "pokedex.txt")
    VOCAB_PATH    = os.path.join(DATA_DIR, "processed", "vocab.json")

    print("=" * 55)
    print("  Tokenizador por caracteres — PokeGPT V0.1")
    print("=" * 55)

    # 1. Cargar corpus
    print(f"\nCargando corpus: {CORPUS_PATH}")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = f.read()
    print(f"Caracteres cargados: {len(corpus):,}")

    # 2. Construir vocabulario
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(corpus)
    print(f"\nVocabulario construido: {tokenizer.vocab_size} tokens")
    print(f"  Índice 0 reservado para: '{CharTokenizer.TOKEN_UNK}'")

    # 3. Mostrar el vocab completo
    print("\n--- Vocabulario completo (índice → carácter) ---")
    for idx in range(tokenizer.vocab_size):
        char = tokenizer.idx2char[idx]
        print(f"  {idx:>3}  {repr(char)}")

    # 4. Guardar en disco
    tokenizer.save(VOCAB_PATH)
    print(f"\nVocabulario guardado en: {VOCAB_PATH}")

    # 5. Prueba de encode → decode (round-trip)
    muestra = corpus[:200]
    encoded = tokenizer.encode(muestra)
    decoded = tokenizer.decode(encoded)

    print("\n--- Prueba round-trip (encode → decode) ---")
    print(f"  Texto original  : {repr(muestra[:80])}")
    print(f"  Encoded (primeros 30 tokens): {encoded[:30]}")
    print(f"  Decoded         : {repr(decoded[:80])}")
    print(f"  Round-trip OK   : {muestra == decoded}")

    # 6. Prueba con texto nuevo (carácter no visto → <UNK>)
    texto_nuevo  = "Pikachu usa Thunderbolt! 🔥"
    enc_nuevo    = tokenizer.encode(texto_nuevo)
    dec_nuevo    = tokenizer.decode(enc_nuevo)
    print("\n--- Prueba con carácter fuera del vocab (emoji) ---")
    print(f"  Input  : {repr(texto_nuevo)}")
    print(f"  Encoded: {enc_nuevo}")
    print(f"  Decoded: {repr(dec_nuevo)}  (<UNK>=índice 0 → '?')")

    # 7. Estadísticas finales
    print("\n--- Estadísticas ---")
    print(f"  Corpus completo encoded: {len(corpus):,} tokens")
    print(f"  Vocab size             : {tokenizer.vocab_size}")
    print(f"  Token mínimo           : 0  ({repr(tokenizer.idx2char[0])})")
    print(f"  Token máximo           : {tokenizer.vocab_size - 1}  ({repr(tokenizer.idx2char[tokenizer.vocab_size - 1])})")

    print("\n[OK] Tokenizador listo.")
    print("=" * 55)
