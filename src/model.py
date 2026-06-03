"""
model.py — Transformer Decoder de PokeGPT, construido desde cero.

Este archivo se construye de forma incremental a lo largo de la Semana 1-2:
    Sáb  — TokenEmbedding + PositionalEncoding        (este archivo, hoy)
    Dom  — SelfAttention (Q, K, V, scaled dot-product)
    Lun  — MultiHeadAttention
    Mar  — FeedForward + residual + LayerNorm
    Mié  — DecoderBlock + PokeGPTModel completo

¿Por qué necesitamos Embedding + Positional Encoding?
    El modelo recibe tensores de índices enteros: [21, 66, 57, ...].
    Esos números no tienen ninguna geometría — el índice 21 no es "cercano"
    al 22 ni "lejano" del 88. Son etiquetas arbitrarias.

    El Embedding convierte cada índice en un vector denso de floats donde
    la geometría SÍ tiene significado: tokens similares acaban cerca en el
    espacio vectorial, y esa geometría se aprende durante el entrenamiento.

    El Positional Encoding añade información sobre la posición de cada token
    en la secuencia. Sin él, el modelo no sabría si "Bulbasaur" aparece al
    principio o al final — trataría la secuencia como una bolsa de tokens.

Archivos relacionados:
    src/tokenizer.py   — convierte texto a índices
    src/dataset.py     — produce batches (batch, seq_len) de índices
    .env               — EMBED_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, CONTEXT_LENGTH
"""

import math
import torch
import torch.nn as nn
from dotenv import load_dotenv
import os

load_dotenv()


# ─── 1. TOKEN EMBEDDING ───────────────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    """
    Tabla de búsqueda aprendible: convierte índices de tokens en vectores densos.

    Internamente es una matriz de pesos de forma (vocab_size, embed_dim).
    Cada fila es el vector representativo de un token.
    Cuando el modelo recibe el índice 21 (la 'B' de Bulbasaur), devuelve
    la fila 21 de esa matriz.

    ¿Por qué nn.Parameter y no nn.Embedding?
        nn.Embedding es exactamente esto pero envuelto en una clase de PyTorch.
        Aquí usamos nn.Parameter directamente para ver que un embedding no es
        más que una matriz de números que se actualiza con backpropagation,
        igual que cualquier otro peso de la red.

    Args:
        vocab_size: número total de tokens (89 en nuestro caso).
        embed_dim:  dimensión del vector por token (128 por defecto en .env).
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()

        # La tabla de embeddings: una fila por token, embed_dim columnas.
        # Inicializamos con valores aleatorios pequeños — el entrenamiento
        # irá ajustando estos valores para que capturen similitudes semánticas.
        self.weight = nn.Parameter(
            torch.randn(vocab_size, embed_dim) * 0.02
        )
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Búsqueda vectorizada en la tabla de embeddings.

        Args:
            indices: tensor de forma (batch_size, seq_len) con índices enteros.

        Returns:
            tensor de forma (batch_size, seq_len, embed_dim) con los vectores
            correspondientes a cada índice.

        Internamente esto es equivalente a una multiplicación one-hot × weight,
        pero PyTorch indexa directamente la fila en lugar de hacer la multiplicación
        completa (mucho más eficiente).
        """
        # self.weight[indices] hace indexación avanzada de PyTorch:
        # para cada índice en el tensor, extrae la fila correspondiente de weight.
        return self.weight[indices]


# ─── 2. POSITIONAL ENCODING ───────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal (fija, no aprendida).

    ¿Por qué necesitamos posición?
        La operación de atención es invariante al orden: si mezclamos los tokens
        de la secuencia, la atención produce el mismo resultado. El modelo no
        sabría que "Bulbasaur" va antes que "es". El positional encoding
        inyecta información de posición en cada vector.

    ¿Por qué sinusoidal y no aprendida?
        Las funciones seno y coseno tienen propiedades matemáticas útiles:
        - Son únicas por posición (ninguna posición tiene el mismo patrón).
        - La diferencia entre dos posiciones es codificable mediante rotaciones
          lineales, lo que facilita que el modelo aprenda distancias relativas.
        - Funcionan para secuencias más largas que las vistas en entrenamiento.

    La fórmula (del paper "Attention Is All You Need", Vaswani et al. 2017):
        PE(pos, 2i)   = sin(pos / 10000^(2i / embed_dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / embed_dim))

        pos  = posición del token en la secuencia (0, 1, 2, ...)
        i    = índice de la dimensión del embedding (0, 1, ..., embed_dim/2)

        Las dimensiones pares usan seno, las impares coseno.
        La frecuencia disminuye con i: las primeras dimensiones oscilan rápido
        (distinguen posiciones cercanas), las últimas oscilan lento
        (distinguen posiciones lejanas).

    Args:
        embed_dim:   dimensión del embedding (debe coincidir con TokenEmbedding).
        max_len:     longitud máxima de secuencia que soporta el modelo.
        dropout:     probabilidad de dropout aplicado tras sumar la codificación.
    """

    def __init__(self, embed_dim: int, max_len: int, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Construimos la tabla de codificaciones posicionales.
        # pe[pos, dim] = sin o cos según la fórmula anterior.
        # La calculamos una sola vez en el constructor y la registramos como
        # buffer: no es un parámetro aprendible, pero sí viaja con el modelo
        # al guardar/cargar pesos y se mueve a GPU si hace falta.
        pe = torch.zeros(max_len, embed_dim)   # (max_len, embed_dim)

        # Vector de posiciones: [0, 1, 2, ..., max_len-1], forma (max_len, 1)
        posicion = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Factor de escala para cada par de dimensiones.
        # Equivale a 1 / 10000^(2i/embed_dim), calculado en log-space
        # para estabilidad numérica.
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )

        # Dimensiones pares → seno, dimensiones impares → coseno
        pe[:, 0::2] = torch.sin(posicion * div_term)
        pe[:, 1::2] = torch.cos(posicion * div_term)

        # Añadimos dimensión de batch para que sea compatible con (batch, seq, dim)
        pe = pe.unsqueeze(0)   # (1, max_len, embed_dim)

        # register_buffer: PyTorch lo incluye en state_dict (guardado/cargado)
        # pero NO lo trata como parámetro aprendible.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Suma la codificación posicional al tensor de embeddings.

        Args:
            x: embeddings de tokens, forma (batch_size, seq_len, embed_dim).

        Returns:
            tensor de la misma forma con la información posicional incorporada.
        """
        # x.size(1) es la longitud real de la secuencia.
        # Cortamos pe hasta esa longitud por si la secuencia es más corta que max_len.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─── 3. EMBEDDING COMBINADO ───────────────────────────────────────────────────

class InputEmbedding(nn.Module):
    """
    Combina TokenEmbedding y PositionalEncoding en un único módulo.

    Este es el primer bloque del Transformer: transforma un batch de índices
    enteros en vectores flotantes enriquecidos con información posicional,
    listos para entrar en los bloques de atención.

    También aplica el factor de escala sqrt(embed_dim) recomendado en el
    paper original. Sin él, los embeddings de tokens tendrían magnitud mucho
    menor que el positional encoding y este dominaría la señal.

    Args:
        vocab_size:  tamaño del vocabulario.
        embed_dim:   dimensión de los vectores.
        max_len:     longitud máxima de secuencia.
        dropout:     dropout aplicado tras el positional encoding.
    """

    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_enc   = PositionalEncoding(embed_dim, max_len, dropout)
        self.embed_dim = embed_dim

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (batch_size, seq_len) — índices de tokens enteros.

        Returns:
            (batch_size, seq_len, embed_dim) — vectores listos para la atención.
        """
        # Escalamos los embeddings por sqrt(embed_dim) antes de sumar la posición
        tokens = self.token_emb(indices) * math.sqrt(self.embed_dim)
        return self.pos_enc(tokens)


# ─── Script de prueba ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Verifica que TokenEmbedding y PositionalEncoding funcionan:
      1. Shapes correctas en cada etapa.
      2. El positional encoding es diferente para cada posición.
      3. El embedding combinado produce la salida esperada.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from tokenizer import CharTokenizer
    from dotenv import load_dotenv
    load_dotenv()

    VOCAB_SIZE     = 89
    EMBED_DIM      = int(os.getenv("EMBED_DIM",      128))
    CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", 128))
    DROPOUT        = float(os.getenv("DROPOUT",      0.1))
    BATCH_SIZE     = 4   # pequeño para la demo

    print("=" * 55)
    print("  Embedding + Positional Encoding — PokeGPT V0.1")
    print("=" * 55)

    # Batch de ejemplo: 4 secuencias de 16 tokens (índices aleatorios)
    torch.manual_seed(42)
    indices_demo = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, 16))
    print(f"\nEntrada (indices):  shape={indices_demo.shape}  dtype={indices_demo.dtype}")
    print(f"  Ejemplo fila 0: {indices_demo[0].tolist()}")

    # ── TokenEmbedding ──────────────────────────────────────────────────────
    print("\n--- TokenEmbedding ---")
    token_emb = TokenEmbedding(VOCAB_SIZE, EMBED_DIM)
    embeddings = token_emb(indices_demo)
    print(f"  Tabla de pesos shape : {token_emb.weight.shape}  → (vocab_size={VOCAB_SIZE}, embed_dim={EMBED_DIM})")
    print(f"  Salida shape         : {embeddings.shape}  → (batch={BATCH_SIZE}, seq=16, embed_dim={EMBED_DIM})")
    print(f"  Vector del token 0   : {embeddings[0, 0, :8].tolist()} ...")
    print(f"  (Son los primeros 8 de {EMBED_DIM} valores del vector del primer token)")

    # ── PositionalEncoding ──────────────────────────────────────────────────
    print("\n--- PositionalEncoding ---")
    pos_enc = PositionalEncoding(EMBED_DIM, CONTEXT_LENGTH, dropout=0.0)
    # Desactivamos dropout para ver los valores exactos
    pos_enc.eval()
    pe_tabla = pos_enc.pe[0]  # (max_len, embed_dim)
    print(f"  Tabla PE shape       : {pe_tabla.shape}  → (max_len={CONTEXT_LENGTH}, embed_dim={EMBED_DIM})")
    print(f"  PE posición 0 (primeros 8): {pe_tabla[0, :8].tolist()}")
    print(f"  PE posición 1 (primeros 8): {pe_tabla[1, :8].tolist()}")
    print(f"  PE posición 2 (primeros 8): {pe_tabla[2, :8].tolist()}")
    print(f"  (Cada posición tiene un patron unico de senos y cosenos)")

    # Verificar que no hay dos posiciones iguales
    difs = (pe_tabla[0] - pe_tabla[1]).abs().sum().item()
    print(f"  Diferencia total pos 0 vs pos 1: {difs:.4f}  (debe ser > 0)")

    # ── InputEmbedding combinado ─────────────────────────────────────────────
    print("\n--- InputEmbedding (Token + Posicion combinados) ---")
    input_emb = InputEmbedding(VOCAB_SIZE, EMBED_DIM, CONTEXT_LENGTH, DROPOUT)
    input_emb.eval()
    salida = input_emb(indices_demo)
    print(f"  Entrada shape : {indices_demo.shape}")
    print(f"  Salida shape  : {salida.shape}  → (batch={BATCH_SIZE}, seq=16, embed_dim={EMBED_DIM})")
    print(f"  dtype         : {salida.dtype}")
    print(f"  Valores finitos (sin NaN/Inf): {torch.isfinite(salida).all().item()}")

    # Parámetros aprendibles
    params = sum(p.numel() for p in input_emb.parameters())
    print(f"\n  Parametros aprendibles: {params:,}")
    print(f"  (Solo la tabla TokenEmbedding: {VOCAB_SIZE} x {EMBED_DIM} = {VOCAB_SIZE*EMBED_DIM:,})")
    print(f"  (El PositionalEncoding NO tiene parametros — es fijo)")

    print("\n[OK] Embedding y Positional Encoding listos.")
    print("=" * 55)
