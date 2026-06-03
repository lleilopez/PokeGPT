"""
model.py — Transformer Decoder de PokeGPT, construido desde cero.

Este archivo se construye de forma incremental a lo largo de la Semana 1-2:
    Sáb  — TokenEmbedding + PositionalEncoding        [HECHO]
    Dom  — SelfAttention (Q, K, V, scaled dot-product) [HECHO]
    Lun  — MultiHeadAttention                              [HECHO]
    Mar  — FeedForward + residual + LayerNorm              [HECHO]
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


# ─── 6. MULTI-HEAD ATTENTION ─────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Atención multi-cabeza: ejecuta varias atenciones en paralelo y combina resultados.

    ¿Por qué múltiples cabezas?
        Una sola cabeza de atención aprende UNA forma de relacionar tokens entre sí.
        Con varias cabezas, el modelo puede aprender SIMULTÁNEAMENTE distintos tipos
        de relaciones en la misma secuencia.

        Ejemplo en un texto de Pokémon:
            Cabeza 1 → aprende relaciones de tipo ("Planta" ↔ "Veneno")
            Cabeza 2 → aprende relaciones de stats ("HP alto" ↔ "defensor")
            Cabeza 3 → aprende relaciones posicionales (sujeto ↔ verbo)
            Cabeza 4 → aprende relaciones de movimientos (ataque ↔ cobertura)

        Ninguna cabeza ve la información completa: cada una trabaja con un
        subespacio de dimensión embed_dim / num_heads. Al final se concatenan
        y se proyectan de vuelta a embed_dim.

    ¿Cómo se implementa sin bucles?
        En lugar de crear num_heads objetos SelfAttention separados, hacemos
        una sola operación matricial y luego reordenamos las dimensiones.

        El truco está en el reshape: si embed_dim=128 y num_heads=4,
        cada cabeza trabaja con d_head = 128/4 = 32 dimensiones.

        Reshape de (batch, seq, embed_dim) → (batch, num_heads, seq, d_head)
        Así PyTorch aplica la atención en paralelo sobre las num_heads cabezas.

    Flujo completo:
        x (batch, seq, embed_dim)
        → proyectar W_q, W_k, W_v  → (batch, seq, embed_dim)
        → split en cabezas          → (batch, num_heads, seq, d_head)
        → scaled dot-product attn   → (batch, num_heads, seq, d_head)
        → concatenar cabezas        → (batch, seq, embed_dim)
        → proyección final W_o      → (batch, seq, embed_dim)

    Args:
        embed_dim:  dimensión total de los embeddings. Debe ser divisible por num_heads.
        num_heads:  número de cabezas de atención.
        dropout:    dropout aplicado sobre los pesos de atención.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) debe ser divisible por num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head    = embed_dim // num_heads   # dimensión por cabeza

        # Una sola proyección grande para Q, K y V.
        # Es matemáticamente equivalente a tener num_heads proyecciones pequeñas,
        # pero más eficiente porque es una sola operación matricial.
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Proyección final: combina las salidas de todas las cabezas
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide el tensor en num_heads cabezas reordenando dimensiones.

        Transforma (batch, seq_len, embed_dim)
                 → (batch, num_heads, seq_len, d_head)

        Cada cabeza recibe una "rodaja" de d_head dimensiones del embedding.
        """
        batch, seq_len, _ = x.shape
        # Reshape: (batch, seq, embed) → (batch, seq, num_heads, d_head)
        x = x.view(batch, seq_len, self.num_heads, self.d_head)
        # Transponer: (batch, seq, num_heads, d_head) → (batch, num_heads, seq, d_head)
        # Necesario para que la atención opere sobre la dimensión seq correctamente
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Operación inversa a _split_heads: reúne las cabezas en un solo tensor.

        Transforma (batch, num_heads, seq_len, d_head)
                 → (batch, seq_len, embed_dim)
        """
        batch, _, seq_len, _ = x.shape
        # Deshacer la transposición y volver a fusionar las dimensiones de cabeza
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.embed_dim)

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, embed_dim)
            mask: máscara causal (1, seq_len, seq_len) o None

        Returns:
            (batch, seq_len, embed_dim)
        """
        # 1. Proyectar a Q, K, V — shape: (batch, seq_len, embed_dim)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 2. Dividir en cabezas — shape: (batch, num_heads, seq_len, d_head)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 3. Atención en paralelo sobre todas las cabezas
        # La máscara necesita una dimensión extra para num_heads: (1, 1, seq, seq)
        if mask is not None:
            mask = mask.unsqueeze(1)   # (1, seq, seq) → (1, 1, seq, seq)

        attn_out, self.attn_weights = scaled_dot_product_attention(q, k, v, mask)
        # attn_out shape: (batch, num_heads, seq_len, d_head)

        # 4. Reunir cabezas — shape: (batch, seq_len, embed_dim)
        attn_out = self._merge_heads(attn_out)

        # 5. Proyección final + dropout
        return self.W_o(self.dropout(attn_out))


# ─── 7. FEED-FORWARD ─────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Red feed-forward aplicada posición a posición tras la atención.

    ¿Qué hace y por qué existe?
        La atención mezcla información entre tokens (comunicación entre posiciones).
        El FeedForward procesa cada token de forma independiente (sin mirar a los
        vecinos), permitiendo al modelo hacer transformaciones más complejas sobre
        la representación de cada token individualmente.

        Se puede pensar como: la atención decide QUÉ información recoger de otros
        tokens; el FeedForward decide QUÉ HACER con esa información una vez recogida.

    Arquitectura: dos capas lineales con expansión intermedia
        Linear(embed_dim → 4 * embed_dim)  — proyección a espacio mayor
        GELU()                              — activación no lineal
        Dropout
        Linear(4 * embed_dim → embed_dim)  — proyección de vuelta
        Dropout

    ¿Por qué expandir a 4 * embed_dim?
        El factor 4 viene del paper original (Vaswani et al., 2017) y ha demostrado
        funcionar bien empíricamente. El espacio mayor permite representar
        combinaciones más complejas antes de comprimir de vuelta.
        Con embed_dim=128: 128 → 512 → 128.

    ¿Por qué GELU en vez de ReLU?
        GELU (Gaussian Error Linear Unit) es la activación estándar en LLMs modernos
        (GPT-2, GPT-3, BERT). Es similar a ReLU pero suave en torno al 0,
        lo que estabiliza los gradientes y suele dar mejores resultados en lenguaje.

    Args:
        embed_dim:  dimensión de entrada y salida.
        ff_dim:     dimensión de la capa intermedia (por defecto 4 * embed_dim).
        dropout:    dropout aplicado tras cada capa lineal.
    """

    def __init__(self, embed_dim: int, ff_dim: int | None = None, dropout: float = 0.1):
        super().__init__()

        if ff_dim is None:
            ff_dim = 4 * embed_dim   # expansión estándar del paper original

        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, embed_dim) — misma shape, cada token transformado
        """
        return self.net(x)


# ─── 8. RESIDUAL CONNECTION + LAYER NORM ─────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Envuelve cualquier sub-capa añadiendo conexión residual y Layer Normalization.

    Estos dos mecanismos son esenciales para entrenar redes profundas con éxito.

    CONEXIÓN RESIDUAL (skip connection):
        En lugar de aprender output = F(x), aprendemos output = x + F(x).
        Esto significa que F(x) solo necesita aprender la DIFERENCIA respecto
        a la entrada — lo que "falta" o "hay que corregir".

        Ventaja crítica: el gradiente puede fluir directamente desde la salida
        hasta la entrada sin pasar por F. En redes de muchas capas, sin residual
        el gradiente se multiplica muchas veces y tiende a desvanecerse (vanishing
        gradient). La conexión residual crea un "camino de autopista" para el gradiente.

    LAYER NORMALIZATION:
        Normaliza los valores de cada token individualmente (no por batch).
        Para cada token calcula media y varianza sobre sus embed_dim valores
        y normaliza: x̂ = (x - media) / sqrt(varianza + epsilon).
        Luego aplica parámetros aprendibles gamma (escala) y beta (sesgo).

        ¿Por qué no Batch Normalization?
            BatchNorm normaliza por batch — depende de tener batches grandes y
            se comporta diferente en train vs inference. Para secuencias de longitud
            variable y texto, LayerNorm es más estable y predecible.

    ORDEN PRE-NORM (usado aquí):
        output = x + SubCapa(LayerNorm(x))

        El paper original usaba Post-Norm: LayerNorm(x + SubCapa(x))
        Los LLMs modernos (GPT-2 en adelante) usan Pre-Norm porque es más
        estable durante el entrenamiento, especialmente con muchas capas.

    Args:
        embed_dim: dimensión de los embeddings (para LayerNorm).
        subcapa:   el módulo a envolver (MultiHeadAttention o FeedForward).
        dropout:   dropout adicional sobre la salida de la subcapa.
    """

    def __init__(self, embed_dim: int, subcapa: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.norm    = nn.LayerNorm(embed_dim)
        self.subcapa = subcapa
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Aplica Pre-Norm + subcapa + conexión residual.

        **kwargs permite pasar argumentos extra a la subcapa (por ejemplo mask
        para MultiHeadAttention) sin que ResidualBlock necesite conocerlos.

        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, embed_dim)
        """
        # Pre-Norm: normalizar antes de pasar por la subcapa
        normed = self.norm(x)
        # Aplicar subcapa (atención o feed-forward)
        out    = self.subcapa(normed, **kwargs)
        # Conexión residual: sumar la entrada original
        return x + self.dropout(out)


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

    # ── MultiHeadAttention ───────────────────────────────────────────────────
    print("\n--- MultiHeadAttention ---")
    NUM_HEADS = int(os.getenv("NUM_HEADS", 4))
    D_HEAD    = EMBED_DIM // NUM_HEADS

    mha = MultiHeadAttention(EMBED_DIM, NUM_HEADS, dropout=0.0)
    mha.eval()

    mask_demo2 = crear_mascara_causal(SEQ_LEN, device=torch.device("cpu"))
    salida_mha = mha(x_demo, mask=mask_demo2)

    print(f"  num_heads : {NUM_HEADS}")
    print(f"  d_head    : {D_HEAD}  (embed_dim {EMBED_DIM} / num_heads {NUM_HEADS})")
    print(f"  Entrada shape      : {x_demo.shape}")
    print(f"  Salida shape       : {salida_mha.shape}  (misma que la entrada)")
    print(f"  attn_weights shape : {mha.attn_weights.shape}  → (batch=4, heads={NUM_HEADS}, seq={SEQ_LEN}, seq={SEQ_LEN})")

    # Verificar máscara en cada cabeza
    print(f"\n  Pesos token 0, cabeza 0: {[round(p,4) for p in mha.attn_weights[0,0,0].tolist()]}")
    print(f"  Pesos token 0, cabeza 1: {[round(p,4) for p in mha.attn_weights[0,1,0].tolist()]}")
    print(f"  (Token 0 tiene peso 1.0 en posición 0 en todas las cabezas)")

    print(f"\n  Pesos token 3, cabeza 0: {[round(p,4) for p in mha.attn_weights[0,0,3].tolist()]}")
    print(f"  Pesos token 3, cabeza 1: {[round(p,4) for p in mha.attn_weights[0,1,3].tolist()]}")
    print(f"  (Cada cabeza distribuye la atención de forma distinta sobre tokens 0-3)")

    params_mha = sum(p.numel() for p in mha.parameters())
    print(f"\n  Parámetros aprendibles : {params_mha:,}")
    print(f"  Mismos que SelfAttention: 4 matrices {EMBED_DIM}x{EMBED_DIM}")
    print(f"  La diferencia es cómo se usan: {NUM_HEADS} cabezas de {D_HEAD} dims cada una")

    # Comparar: SelfAttention vs MultiHeadAttention tienen mismos params
    assert params_mha == params_attn, "Deben tener los mismos parámetros"
    print(f"  SelfAttention params == MultiHeadAttention params: {params_mha == params_attn}")

    print("\n[OK] MultiHeadAttention lista.")

    # ── FeedForward ──────────────────────────────────────────────────────────
    print("\n--- FeedForward ---")
    FF_DIM = 4 * EMBED_DIM

    ff = FeedForward(EMBED_DIM, dropout=0.0)
    ff.eval()
    salida_ff = ff(x_demo)

    print(f"  Entrada shape  : {x_demo.shape}")
    print(f"  Salida shape   : {salida_ff.shape}  (misma que la entrada)")
    print(f"  Expansión interna: {EMBED_DIM} → {FF_DIM} → {EMBED_DIM}")

    params_ff = sum(p.numel() for p in ff.parameters())
    print(f"  Parámetros     : {params_ff:,}")
    print(f"  ({EMBED_DIM}x{FF_DIM} + {FF_DIM}) + ({FF_DIM}x{EMBED_DIM} + {EMBED_DIM}) = {params_ff:,}")

    # ── ResidualBlock con FeedForward ────────────────────────────────────────
    print("\n--- ResidualBlock (LayerNorm + FeedForward + residual) ---")
    ff2      = FeedForward(EMBED_DIM, dropout=0.0)
    res_ff   = ResidualBlock(EMBED_DIM, ff2, dropout=0.0)
    res_ff.eval()

    salida_res = res_ff(x_demo)
    print(f"  Entrada shape  : {x_demo.shape}")
    print(f"  Salida shape   : {salida_res.shape}")

    # Verificar que la conexión residual funciona:
    # la salida NO es igual a la entrada (la subcapa añade algo)
    # pero tampoco difiere en exceso (la residual estabiliza)
    diff = (salida_res - x_demo).abs().mean().item()
    print(f"  Diferencia media entrada/salida: {diff:.6f}  (> 0 confirma que FF transforma)")

    # Verificar que LayerNorm normaliza correctamente
    x_normed = res_ff.norm(x_demo)
    print(f"  Media tras LayerNorm  : {x_normed.mean().item():.6f}  (debe ser ~0)")
    print(f"  Std  tras LayerNorm   : {x_normed.std().item():.6f}   (debe ser ~1)")

    # ── ResidualBlock con MultiHeadAttention ────────────────────────────────
    print("\n--- ResidualBlock (LayerNorm + MultiHeadAttention + residual) ---")
    mha2    = MultiHeadAttention(EMBED_DIM, NUM_HEADS, dropout=0.0)
    res_mha = ResidualBlock(EMBED_DIM, mha2, dropout=0.0)
    res_mha.eval()

    salida_res_mha = res_mha(x_demo, mask=mask_demo2)
    print(f"  Entrada shape  : {x_demo.shape}")
    print(f"  Salida shape   : {salida_res_mha.shape}")
    print(f"  Valores finitos: {torch.isfinite(salida_res_mha).all().item()}")

    # ── Resumen de parámetros acumulados ────────────────────────────────────
    print("\n--- Resumen de parámetros por bloque ---")
    print(f"  InputEmbedding     : {sum(p.numel() for p in input_emb.parameters()):>8,}")
    print(f"  MultiHeadAttention : {sum(p.numel() for p in mha2.parameters()):>8,}")
    print(f"  FeedForward        : {sum(p.numel() for p in ff2.parameters()):>8,}")
    print(f"  LayerNorm (x2)     : {sum(p.numel() for p in res_ff.norm.parameters()) * 2:>8,}  (2 params por dim × 2 norms)")
    total = (sum(p.numel() for p in input_emb.parameters()) +
             sum(p.numel() for p in mha2.parameters()) +
             sum(p.numel() for p in ff2.parameters()) +
             sum(p.numel() for p in res_ff.norm.parameters()) * 2)
    print(f"  ─────────────────────────────")
    print(f"  Total 1 bloque     : {total:>8,}  (sin contar la capa de salida)")

    print("\n[OK] FeedForward + Residual + LayerNorm listos.")
    print("=" * 55)
