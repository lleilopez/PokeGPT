"""
model.py — Transformer Decoder de PokeGPT, construido desde cero.

Este archivo se construye de forma incremental a lo largo de la Semana 1-2:
    Sáb  — TokenEmbedding + PositionalEncoding        [HECHO]
    Dom  — SelfAttention (Q, K, V, scaled dot-product) [HECHO]
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


# ─── 4. SCALED DOT-PRODUCT ATTENTION ─────────────────────────────────────────

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    La operación fundamental de atención del Transformer.

    Intuición: imagina que cada token de la secuencia hace una "pregunta" (Query)
    y todos los tokens ofrecen una "clave" (Key). La similitud entre la pregunta
    y cada clave determina cuánta "atención" se le presta a cada token. El
    resultado es una suma ponderada de los "valores" (Value) de todos los tokens.

    En un modelo de lenguaje:
        Q = "¿qué información necesito para predecir el siguiente token?"
        K = "¿qué información tengo yo para ofrecer?"
        V = "esta es la información que ofrezco si me seleccionan"

    Fórmula (Vaswani et al., 2017):
        Attention(Q, K, V) = softmax( Q·K^T / sqrt(d_k) ) · V

    ¿Por qué dividir por sqrt(d_k)?
        El producto punto Q·K^T crece en magnitud con d_k (dimensión de las claves).
        Con valores muy grandes, la softmax se satura: un valor domina con ~1.0 y
        los demás caen a ~0.0. El gradiente se vuelve casi cero y el modelo deja
        de aprender. Dividir por sqrt(d_k) mantiene la varianza estable.

    ¿Por qué necesitamos mask (máscara causal)?
        En un modelo autoregresivo, el token en la posición i solo puede ver los
        tokens 0..i-1. Si no enmascaramos, el token en posición 3 "vería" el
        token en posición 4, que es justamente lo que tiene que predecir.
        Sería como hacer un examen con las respuestas delante.

        La máscara añade -inf a las posiciones futuras ANTES de softmax.
        softmax(-inf) = 0, así que esas posiciones reciben atención cero.

        Visualmente para una secuencia de longitud 4:
            posición → puede ver:
            0        → [0]           (solo se ve a sí misma)
            1        → [0, 1]
            2        → [0, 1, 2]
            3        → [0, 1, 2, 3]

    Args:
        q:    Queries,  shape (..., seq_len, d_k)
        k:    Keys,     shape (..., seq_len, d_k)
        v:    Values,   shape (..., seq_len, d_v)
        mask: máscara booleana o de floats, shape (..., seq_len, seq_len).
              Las posiciones con True (o -inf) se enmascaran.

    Returns:
        output:  tensor ponderado, shape (..., seq_len, d_v)
        weights: pesos de atención tras softmax, shape (..., seq_len, seq_len)
                 útil para visualizar qué tokens se atienden entre sí.
    """
    d_k = q.size(-1)   # dimensión de las claves

    # Paso 1: similitud entre queries y keys → scores (..., seq_len, seq_len)
    # scores[i, j] = cuánto se parece la query del token i a la key del token j
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Paso 2: aplicar máscara causal (si existe)
    # Rellenamos con un valor muy negativo las posiciones futuras
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Paso 3: softmax sobre la última dimensión → pesos que suman 1 por fila
    # Cada token distribuye el 100% de su atención entre los tokens permitidos
    weights = torch.softmax(scores, dim=-1)

    # Paso 4: suma ponderada de values
    output = torch.matmul(weights, v)

    return output, weights


def crear_mascara_causal(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Construye la máscara causal (triangular superior) para el decoder.

    Devuelve un tensor booleano donde True indica "esta posición debe ser
    enmascarada" (no puede ser atendida porque está en el futuro).

    Ejemplo para seq_len=4:
        [[False, True,  True,  True ],
         [False, False, True,  True ],
         [False, False, False, True ],
         [False, False, False, False]]

    Args:
        seq_len: longitud de la secuencia.
        device:  dispositivo donde crear el tensor (cpu o cuda).

    Returns:
        Tensor booleano de forma (1, seq_len, seq_len).
    """
    # torch.ones crea una matriz de unos, triu extrae el triángulo superior
    # diagonal=1 excluye la diagonal principal (un token SÍ puede atenderse a sí mismo)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0)   # añadimos dimensión de batch: (1, seq, seq)


# ─── 5. SELF-ATTENTION (una sola cabeza) ──────────────────────────────────────

class SelfAttention(nn.Module):
    """
    Bloque de auto-atención de una sola cabeza.

    "Auto" significa que Q, K y V se generan todos a partir de la misma
    entrada x. El bloque aprende a relacionar cada token con los demás
    tokens de la misma secuencia.

    Proceso:
        1. Proyectar x → Q, K, V mediante tres capas lineales independientes.
           Cada proyección puede aprender a "extraer" distintos aspectos del token.
        2. Aplicar scaled dot-product attention con máscara causal.
        3. Proyectar la salida de vuelta a embed_dim (proyección de salida).

    ¿Por qué proyecciones lineales y no usar x directamente como Q, K, V?
        Si usáramos x directamente, Q = K = V = x, y la atención aprendería
        una sola "forma de relacionarse". Las proyecciones dan al modelo la
        libertad de aprender representaciones diferentes para preguntar (Q),
        para ofrecer claves (K) y para ofrecer valores (V).

    Args:
        embed_dim: dimensión de los vectores de entrada y salida.
        dropout:   dropout aplicado sobre los pesos de atención.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim

        # Tres proyecciones lineales: una para Q, otra para K, otra para V.
        # bias=False es común en implementaciones modernas de atención.
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Proyección de salida: combina la información de atención
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    embeddings de entrada, shape (batch, seq_len, embed_dim).
            mask: máscara causal, shape (1, seq_len, seq_len) o None.

        Returns:
            tensor de la misma shape que x: (batch, seq_len, embed_dim).
        """
        # Proyectar x a Q, K, V — cada uno tiene shape (batch, seq_len, embed_dim)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Aplicar la atención con máscara causal
        attn_output, self.attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # Guardamos attn_weights como atributo para poder inspeccionarlos
        # externamente si queremos visualizar qué tokens se atienden.

        # Aplicar dropout sobre la salida y proyectar al espacio original
        attn_output = self.dropout(attn_output)
        return self.W_o(attn_output)


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

    # ── Máscara causal ───────────────────────────────────────────────────────
    print("\n--- Máscara causal (seq_len=6) ---")
    mask = crear_mascara_causal(6, device=torch.device("cpu"))
    print(f"  Shape: {mask.shape}")
    print("  Matriz (True = posición enmascarada = futuro):")
    for fila in mask[0]:
        print("    " + "  ".join(["T" if v else "." for v in fila.tolist()]))
    print("  (. = puede ver este token, T = no puede — está en el futuro)")

    # ── SelfAttention ────────────────────────────────────────────────────────
    print("\n--- SelfAttention (una cabeza) ---")
    SEQ_LEN = 16

    # Simulamos la salida del InputEmbedding: (batch=4, seq=16, embed=128)
    x_demo  = input_emb(indices_demo)
    mask_demo = crear_mascara_causal(SEQ_LEN, device=torch.device("cpu"))

    attn = SelfAttention(EMBED_DIM, dropout=0.0)
    attn.eval()
    salida_attn = attn(x_demo, mask=mask_demo)

    print(f"  Entrada shape : {x_demo.shape}")
    print(f"  Salida shape  : {salida_attn.shape}  (misma shape que la entrada)")
    print(f"  Pesos de atención shape: {attn.attn_weights.shape}  → (batch=4, seq={SEQ_LEN}, seq={SEQ_LEN})")

    # Verificar que la máscara funciona: los pesos del futuro deben ser 0
    # En la fila 0 (primer token), solo la posición 0 puede tener peso > 0
    pesos_fila0 = attn.attn_weights[0, 0]   # pesos del token 0 en la secuencia 0
    print(f"\n  Pesos de atención del token 0 (solo puede verse a sí mismo):")
    print(f"    {[round(p, 4) for p in pesos_fila0.tolist()]}")
    print(f"    (el token 0 tiene peso 1.0 en posición 0 y 0.0 en el resto)")

    pesos_fila3 = attn.attn_weights[0, 3]   # pesos del token 3
    print(f"\n  Pesos de atención del token 3 (puede ver tokens 0,1,2,3):")
    print(f"    {[round(p, 4) for p in pesos_fila3.tolist()]}")
    print(f"    (los 4 primeros suman 1.0, el resto son 0.0)")

    params_attn = sum(p.numel() for p in attn.parameters())
    print(f"\n  Parámetros aprendibles: {params_attn:,}")
    print(f"  (4 matrices lineales de {EMBED_DIM}x{EMBED_DIM}: W_q, W_k, W_v, W_o)")

    print("\n[OK] SelfAttention lista.")
    print("=" * 55)
