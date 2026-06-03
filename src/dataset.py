"""
dataset.py — Convierte el corpus tokenizado en tensores listos para entrenar.

¿Qué hace este módulo?
    1. Carga el corpus de texto crudo.
    2. Lo codifica completo en un único tensor de enteros (un número por carácter).
    3. Expone ese tensor como un Dataset de PyTorch que produce pares (input, target).
    4. Divide el corpus en split de entrenamiento y validación.
    5. Crea los DataLoaders que el bucle de entrenamiento consumirá.

¿Por qué un único tensor largo?
    Convertir todo el corpus a un solo tensor 1D es eficiente en memoria.
    Cada __getitem__ simplemente corta una ventana de ese tensor, sin copiar datos
    hasta que PyTorch los necesita para construir el batch.

¿Cómo funciona la ventana deslizante?
    Para entrenar un modelo autoregresivo necesitamos pares (pregunta, respuesta):
    dado el contexto anterior, predecir el siguiente token.

    Ejemplo con context_length=5:
        corpus:  [B, u, l, b, a, s, a, u, r, ...]
        índice 0:  input=[B,u,l,b,a]   target=[u,l,b,a,s]
        índice 1:  input=[u,l,b,a,s]   target=[l,b,a,s,a]
        índice 2:  input=[l,b,a,s,a]   target=[b,a,s,a,u]
        ...

    El target es el input desplazado 1 posición a la derecha.
    En cada paso el modelo aprende: "dado este contexto, el siguiente token es X".

Archivos relacionados:
    data/raw/pokedex.txt          — corpus fuente
    data/processed/vocab.json     — vocabulario del tokenizador
    src/tokenizer.py              — clase CharTokenizer
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from dotenv import load_dotenv

from tokenizer import CharTokenizer

load_dotenv()


class PokeDataset(Dataset):
    """
    Dataset de lenguaje para entrenamiento autoregresivo por caracteres.

    Almacena el corpus completo como un único tensor 1D de enteros (LongTensor).
    Cada elemento devuelto es un par (input, target) de longitud context_length.

    Args:
        data:           tensor 1D con todos los tokens del corpus.
        context_length: número de tokens que el modelo ve como contexto.
    """

    def __init__(self, data: torch.Tensor, context_length: int):
        self.data           = data
        self.context_length = context_length

    def __len__(self) -> int:
        """
        Número de secuencias posibles.

        Con un corpus de N tokens y context_length=C, podemos empezar
        una ventana en cualquier posición de 0 a N-C-1, lo que da N-C
        secuencias válidas.
        """
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve el par (input, target) que empieza en la posición idx.

        input:  tokens[idx   : idx+context_length]
        target: tokens[idx+1 : idx+context_length+1]

        El target es el input desplazado 1 posición: en cada posición i,
        el modelo debe predecir el token que viene después de input[i].
        """
        x = self.data[idx     : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y


def cargar_corpus_como_tensor(
    corpus_path: str,
    vocab_path:  str,
) -> tuple[torch.Tensor, CharTokenizer]:
    """
    Lee el corpus de texto, lo tokeniza y lo convierte a un LongTensor 1D.

    Los embeddings de PyTorch esperan índices de tipo Long (int64),
    por eso usamos dtype=torch.long.

    Returns:
        data:      LongTensor de forma (N,) con N = número de caracteres del corpus.
        tokenizer: instancia de CharTokenizer ya cargada con el vocab.
    """
    # Cargar vocabulario ya construido por tokenizer.py
    tokenizer = CharTokenizer()
    tokenizer.load(vocab_path)

    # Leer el corpus completo
    with open(corpus_path, "r", encoding="utf-8") as f:
        texto = f.read()

    # Codificar: lista de enteros → tensor 1D de tipo Long
    indices = tokenizer.encode(texto)
    data    = torch.tensor(indices, dtype=torch.long)

    return data, tokenizer


def crear_dataloaders(
    corpus_path:    str,
    vocab_path:     str,
    context_length: int,
    batch_size:     int,
    val_fraccion:   float = 0.1,
    seed:           int   = 42,
) -> tuple[DataLoader, DataLoader, CharTokenizer]:
    """
    Crea los DataLoaders de entrenamiento y validación.

    El corpus se divide cronológicamente: el 90% inicial para train,
    el 10% final para validación. No mezclamos aleatoriamente la división
    para que la validación sea siempre sobre texto que el modelo no ha visto.

    Args:
        corpus_path:    ruta al archivo de texto crudo.
        vocab_path:     ruta al vocab.json del tokenizador.
        context_length: longitud de cada secuencia de entrada.
        batch_size:     número de secuencias por batch.
        val_fraccion:   fracción del corpus para validación (por defecto 0.1 = 10%).
        seed:           semilla para reproducibilidad del DataLoader.

    Returns:
        train_loader:   DataLoader de entrenamiento.
        val_loader:     DataLoader de validación.
        tokenizer:      instancia cargada del tokenizador.
    """
    data, tokenizer = cargar_corpus_como_tensor(corpus_path, vocab_path)

    # Calcular el punto de corte train/val
    n_total    = len(data)
    n_val      = int(n_total * val_fraccion)
    n_train    = n_total - n_val

    # División cronológica: primeros n_train tokens para train
    data_train = data[:n_train]
    data_val   = data[n_train:]

    dataset_train = PokeDataset(data_train, context_length)
    dataset_val   = PokeDataset(data_val,   context_length)

    # shuffle=True en train: el orden de los batches varía cada época
    # shuffle=False en val:  evaluamos siempre en el mismo orden
    train_loader = DataLoader(
        dataset_train,
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = True,   # descarta el último batch si es incompleto
        generator   = torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size = batch_size,
        shuffle    = False,
        drop_last  = True,
    )

    return train_loader, val_loader, tokenizer


# ─── Script de prueba ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Verifica que el dataset y los dataloaders funcionan correctamente:
      1. Carga corpus y tokenizador.
      2. Muestra forma y tipo del tensor completo.
      3. Inspecciona varios pares (input, target) decodificados.
      4. Crea los DataLoaders y comprueba la forma de un batch real.
    """

    DATA_DIR       = os.getenv("DATA_DIR",       "data")
    CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", 128))
    BATCH_SIZE     = int(os.getenv("BATCH_SIZE",      32))

    CORPUS_PATH = os.path.join(DATA_DIR, "raw",       "pokedex.txt")
    VOCAB_PATH  = os.path.join(DATA_DIR, "processed", "vocab.json")

    print("=" * 55)
    print("  Dataset en tensores — PokeGPT V0.1")
    print("=" * 55)

    # 1. Cargar corpus como tensor
    print("\nCargando corpus y tokenizador...")
    data, tokenizer = cargar_corpus_como_tensor(CORPUS_PATH, VOCAB_PATH)

    print(f"  Tensor shape : {data.shape}")
    print(f"  Dtype        : {data.dtype}")
    print(f"  Token mínimo : {data.min().item()}")
    print(f"  Token máximo : {data.max().item()}  (vocab_size={tokenizer.vocab_size})")

    # 2. Inspeccionar pares (input, target)
    dataset = PokeDataset(data, CONTEXT_LENGTH)
    print(f"\n--- Dataset completo ---")
    print(f"  context_length  : {CONTEXT_LENGTH}")
    print(f"  Secuencias total: {len(dataset):,}")

    print(f"\n--- Ejemplo par (input, target) en posición 0 ---")
    x, y = dataset[0]
    print(f"  input  shape: {x.shape}  dtype: {x.dtype}")
    print(f"  target shape: {y.shape}  dtype: {y.dtype}")
    print(f"  input  (primeros 40 tokens): {x[:40].tolist()}")
    print(f"  target (primeros 40 tokens): {y[:40].tolist()}")
    print(f"  input  decodificado: {repr(tokenizer.decode(x[:60].tolist()))}")
    print(f"  target decodificado: {repr(tokenizer.decode(y[:60].tolist()))}")

    print(f"\n  [Nota] target es input desplazado 1 posicion a la derecha.")
    print(f"  El modelo aprende: dado input[i], predecir target[i].")

    # 3. Crear DataLoaders y ver un batch real
    print(f"\n--- DataLoaders (batch_size={BATCH_SIZE}) ---")
    train_loader, val_loader, _ = crear_dataloaders(
        CORPUS_PATH, VOCAB_PATH, CONTEXT_LENGTH, BATCH_SIZE
    )

    print(f"  Batches de entrenamiento: {len(train_loader):,}")
    print(f"  Batches de validacion   : {len(val_loader):,}")

    # Extraer un batch y mostrar su forma
    xb, yb = next(iter(train_loader))
    print(f"\n--- Primer batch ---")
    print(f"  xb shape: {xb.shape}  → (batch_size={xb.shape[0]}, context_length={xb.shape[1]})")
    print(f"  yb shape: {yb.shape}  → (batch_size={yb.shape[0]}, context_length={yb.shape[1]})")
    print(f"  xb[0] decodificado: {repr(tokenizer.decode(xb[0].tolist()))}")

    print("\n[OK] Dataset y DataLoaders listos para entrenar.")
    print("=" * 55)
