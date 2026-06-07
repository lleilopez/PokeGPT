"""
train.py — Bucle de entrenamiento de PokeGPT.

¿Qué ocurre en el entrenamiento?
    El modelo recibe secuencias de tokens y trata de predecir el siguiente
    token en cada posición. Al principio sus predicciones son malas (aleatorias).
    La loss mide cuánto se equivoca. El backpropagation calcula en qué dirección
    hay que mover cada peso para reducir esa loss. El optimizador aplica ese
    movimiento. Repetido miles de veces, el modelo va aprendiendo patrones.

Componentes del bucle:
    1. Forward pass   — el modelo produce logits a partir de los tokens de entrada
    2. Loss           — CrossEntropyLoss mide el error de predicción
    3. Backward pass  — PyTorch calcula los gradientes automáticamente
    4. Gradient clip  — evita que gradientes muy grandes desestabilicen el modelo
    5. Optimizer step — actualiza los pesos en la dirección que reduce la loss
    6. Zero grad      — limpia los gradientes acumulados antes del siguiente batch

Optimizador: AdamW
    Adam ajusta el learning rate por parámetro de forma adaptativa.
    La W de AdamW añade weight decay (regularización L2 desacoplada),
    que penaliza pesos muy grandes y ayuda a generalizar.

Loss: CrossEntropyLoss
    Para cada posición de la secuencia, tenemos:
        - logits: vector de 89 valores (uno por token del vocabulario)
        - target: el índice del token correcto (el que realmente viene después)
    CrossEntropyLoss aplica softmax internamente y calcula -log(prob_token_correcto).
    Si el modelo asigna prob=1.0 al token correcto → loss=0.
    Si asigna prob=0.01 (casi aleatorio) → loss≈4.5 (=-log(0.01)).

Archivos relacionados:
    src/model.py    — PokeGPTModel
    src/dataset.py  — crear_dataloaders
    src/tokenizer.py — CharTokenizer
    .env            — hiperparámetros
    checkpoints/    — pesos guardados
    logs/           — curvas de loss
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

# Añadimos src/ al path para poder importar los módulos del proyecto
sys.path.insert(0, os.path.dirname(__file__))

from model     import PokeGPTModel
from dataset   import crear_dataloaders
from tokenizer import CharTokenizer

load_dotenv()

# ─── Configuración desde .env ─────────────────────────────────────────────────

DATA_DIR        = os.getenv("DATA_DIR",        "data")
CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", "checkpoints")
LOGS_DIR        = os.getenv("LOGS_DIR",        "logs")

VOCAB_SIZE      = 89
EMBED_DIM       = int(os.getenv("EMBED_DIM",       128))
NUM_HEADS       = int(os.getenv("NUM_HEADS",         4))
NUM_LAYERS      = int(os.getenv("NUM_LAYERS",        2))
CONTEXT_LENGTH  = int(os.getenv("CONTEXT_LENGTH",  128))
DROPOUT         = float(os.getenv("DROPOUT",        0.1))

BATCH_SIZE      = int(os.getenv("BATCH_SIZE",       32))
LEARNING_RATE   = float(os.getenv("LEARNING_RATE", 3e-4))
MAX_EPOCHS      = int(os.getenv("MAX_EPOCHS",      100))
SEED            = int(os.getenv("SEED",             42))
DEVICE_STR      = os.getenv("DEVICE", "cpu")

CORPUS_PATH     = os.path.join(DATA_DIR, "raw",       "pokedex.txt")
VOCAB_PATH      = os.path.join(DATA_DIR, "processed", "vocab.json")

# Cada cuántos batches imprimimos la loss durante el entrenamiento
LOG_CADA_N_BATCHES = 200

# ─── Utilidades ───────────────────────────────────────────────────────────────

def fijar_semilla(seed: int) -> None:
    """Fija todas las semillas para resultados reproducibles."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calcular_loss(
    modelo:  PokeGPTModel,
    xb:      torch.Tensor,
    yb:      torch.Tensor,
) -> torch.Tensor:
    """
    Calcula la CrossEntropyLoss para un batch.

    CrossEntropyLoss necesita:
        - input:  (N, vocab_size)  — logits por token
        - target: (N,)             — índice del token correcto

    El modelo produce (batch, seq_len, vocab_size), así que
    aplanamos las dos primeras dimensiones antes de pasar a la loss.

    Args:
        modelo: PokeGPTModel en modo train.
        xb:     tokens de entrada, shape (batch, seq_len).
        yb:     tokens objetivo,   shape (batch, seq_len).

    Returns:
        loss escalar (tensor de 0 dimensiones).
    """
    logits = modelo(xb)                          # (batch, seq_len, vocab_size)
    B, T, V = logits.shape

    # Aplanar para CrossEntropyLoss
    logits_flat  = logits.view(B * T, V)         # (batch*seq_len, vocab_size)
    targets_flat = yb.view(B * T)                # (batch*seq_len,)

    return F.cross_entropy(logits_flat, targets_flat)


# ─── Validación ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluar(
    modelo:     PokeGPTModel,
    val_loader: torch.utils.data.DataLoader,
    device:     torch.device,
    max_batches: int = 50,
) -> float:
    """
    Calcula la loss media sobre el conjunto de validación.

    @torch.no_grad() desactiva el cálculo de gradientes durante la evaluación:
    no necesitamos backpropagation aquí, así que ahorramos memoria y tiempo.

    Usamos solo max_batches batches para que la validación sea rápida.

    Args:
        modelo:      modelo en modo eval.
        val_loader:  DataLoader de validación.
        device:      cpu o cuda.
        max_batches: número máximo de batches a evaluar.

    Returns:
        loss media como float.
    """
    modelo.eval()
    losses = []

    for i, (xb, yb) in enumerate(val_loader):
        if i >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        loss = calcular_loss(modelo, xb, yb)
        losses.append(loss.item())

    modelo.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ─── Checkpoints ──────────────────────────────────────────────────────────────

def guardar_checkpoint(
    modelo:     PokeGPTModel,
    optimizador: torch.optim.Optimizer,
    epoca:      int,
    val_loss:   float,
    nombre:     str = "checkpoint.pt",
) -> None:
    """
    Guarda el estado completo del entrenamiento en disco.

    Guardamos tanto los pesos del modelo como el estado del optimizador.
    Esto permite reanudar el entrenamiento exactamente donde lo dejamos.

    Args:
        modelo:       PokeGPTModel.
        optimizador:  instancia de AdamW.
        epoca:        número de época actual.
        val_loss:     loss de validación en esta época.
        nombre:       nombre del archivo dentro de checkpoints/.
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ruta = os.path.join(CHECKPOINTS_DIR, nombre)

    torch.save({
        "epoca":           epoca,
        "val_loss":        val_loss,
        "model_state":     modelo.state_dict(),
        "optimizer_state": optimizador.state_dict(),
        "config": {
            "vocab_size":     VOCAB_SIZE,
            "embed_dim":      EMBED_DIM,
            "num_heads":      NUM_HEADS,
            "num_layers":     NUM_LAYERS,
            "context_length": CONTEXT_LENGTH,
            "dropout":        DROPOUT,
        },
    }, ruta)


def cargar_checkpoint(
    ruta:        str,
    modelo:      PokeGPTModel,
    optimizador: torch.optim.Optimizer | None = None,
) -> dict:
    """
    Carga un checkpoint guardado previamente.

    Args:
        ruta:        ruta al archivo .pt.
        modelo:      instancia del modelo donde cargar los pesos.
        optimizador: si se proporciona, también carga su estado.

    Returns:
        diccionario con los metadatos del checkpoint (epoca, val_loss, config).
    """
    checkpoint = torch.load(ruta, weights_only=False)
    modelo.load_state_dict(checkpoint["model_state"])
    if optimizador is not None:
        optimizador.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


# ─── Bucle de entrenamiento ───────────────────────────────────────────────────

def entrenar(
    modelo:        PokeGPTModel,
    train_loader:  torch.utils.data.DataLoader,
    val_loader:    torch.utils.data.DataLoader,
    optimizador:   torch.optim.Optimizer,
    device:        torch.device,
    max_epochs:    int,
    epoca_inicio:  int = 0,
) -> None:
    """
    Bucle principal de entrenamiento.

    Por cada época:
        1. Itera sobre todos los batches de entrenamiento.
        2. En cada batch: forward → loss → backward → clip → step.
        3. Al final de la época: evalúa en validación.
        4. Guarda checkpoint si es la mejor loss de validación hasta ahora.
        5. Imprime estadísticas.

    Args:
        modelo:       PokeGPTModel en modo train.
        train_loader: DataLoader de entrenamiento.
        val_loader:   DataLoader de validación.
        optimizador:  AdamW.
        device:       cpu o cuda.
        max_epochs:   número total de épocas a entrenar.
        epoca_inicio: época desde la que continuar (0 si es nuevo entrenamiento).
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, "loss_history.json")

    mejor_val_loss  = float("inf")
    historial_train = []
    historial_val   = []

    # Si existe un historial previo, lo cargamos para continuar
    if os.path.exists(log_path):
        with open(log_path) as f:
            datos = json.load(f)
            historial_train = datos.get("train", [])
            historial_val   = datos.get("val",   [])

    print(f"\nIniciando entrenamiento desde época {epoca_inicio + 1}/{max_epochs}")
    print(f"Batches por época: {len(train_loader):,}")
    print(f"Dispositivo: {device}")
    print("-" * 55)

    for epoca in range(epoca_inicio, max_epochs):
        modelo.train()
        t_inicio    = time.time()
        loss_acum   = 0.0
        n_batches   = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)

            # ── 1. Forward pass ──────────────────────────────────────────────
            loss = calcular_loss(modelo, xb, yb)

            # ── 2. Backward pass (calcular gradientes) ───────────────────────
            optimizador.zero_grad()   # limpiar gradientes del batch anterior
            loss.backward()           # backpropagation

            # ── 3. Gradient clipping ─────────────────────────────────────────
            # Limita la norma máxima de los gradientes a 1.0.
            # Evita el "exploding gradient": gradientes enormes que disparan
            # los pesos a valores absurdos en un solo paso.
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)

            # ── 4. Actualizar pesos ──────────────────────────────────────────
            optimizador.step()

            loss_acum += loss.item()
            n_batches += 1

            # Log parcial cada LOG_CADA_N_BATCHES batches
            if (batch_idx + 1) % LOG_CADA_N_BATCHES == 0:
                loss_media = loss_acum / n_batches
                print(f"  Época {epoca+1:>3} | Batch {batch_idx+1:>5}/{len(train_loader)} "
                      f"| Loss: {loss_media:.4f}")

        # ── Fin de época: métricas ───────────────────────────────────────────
        train_loss = loss_acum / n_batches
        val_loss   = evaluar(modelo, val_loader, device)
        t_total    = time.time() - t_inicio

        # Perplexity: e^loss — más intuitivo que la loss cruda.
        # Una perplexity de 89 significa que el modelo es tan malo como
        # elegir al azar entre los 89 tokens. Queremos que baje mucho.
        train_ppl = math.exp(train_loss)
        val_ppl   = math.exp(val_loss)

        historial_train.append(train_loss)
        historial_val.append(val_loss)

        print(f"\nÉpoca {epoca+1:>3}/{max_epochs} — "
              f"Train loss: {train_loss:.4f} (ppl {train_ppl:.1f}) | "
              f"Val loss: {val_loss:.4f} (ppl {val_ppl:.1f}) | "
              f"Tiempo: {t_total:.1f}s")

        # ── Guardar checkpoint si mejoramos en validación ────────────────────
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            guardar_checkpoint(modelo, optimizador, epoca + 1, val_loss,
                               nombre="best_model.pt")
            print(f"  [Checkpoint] Nuevo mejor modelo guardado (val_loss={val_loss:.4f})")

        # Checkpoint periódico cada 10 épocas
        if (epoca + 1) % 10 == 0:
            guardar_checkpoint(modelo, optimizador, epoca + 1, val_loss,
                               nombre=f"checkpoint_epoca_{epoca+1}.pt")

        # Guardar historial de losses en JSON
        with open(log_path, "w") as f:
            json.dump({"train": historial_train, "val": historial_val}, f, indent=2)

        print()

    print("=" * 55)
    print(f"Entrenamiento completado.")
    print(f"Mejor val_loss: {mejor_val_loss:.4f}  (ppl {math.exp(mejor_val_loss):.1f})")
    print(f"Modelo guardado en: {CHECKPOINTS_DIR}/best_model.pt")
    print("=" * 55)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    fijar_semilla(SEED)
    device = torch.device(DEVICE_STR)

    print("=" * 55)
    print("  PokeGPT V0.1 — Entrenamiento")
    print("=" * 55)
    print(f"\nConfiguración:")
    print(f"  Dispositivo    : {device}")
    print(f"  Épocas         : {MAX_EPOCHS}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Learning rate  : {LEARNING_RATE}")
    print(f"  Context length : {CONTEXT_LENGTH}")
    print(f"  embed_dim      : {EMBED_DIM}")
    print(f"  num_layers     : {NUM_LAYERS}")
    print(f"  num_heads      : {NUM_HEADS}")

    # Crear DataLoaders
    print("\nCargando dataset...")
    train_loader, val_loader, tokenizer = crear_dataloaders(
        CORPUS_PATH, VOCAB_PATH, CONTEXT_LENGTH, BATCH_SIZE
    )
    print(f"  Batches train : {len(train_loader):,}")
    print(f"  Batches val   : {len(val_loader):,}")

    # Crear modelo
    print("\nCreando modelo...")
    modelo = PokeGPTModel(
        vocab_size     = VOCAB_SIZE,
        embed_dim      = EMBED_DIM,
        num_heads      = NUM_HEADS,
        num_layers     = NUM_LAYERS,
        context_length = CONTEXT_LENGTH,
        dropout        = DROPOUT,
    ).to(device)
    print(f"  Parámetros totales: {modelo.num_params():,}")

    # Crear optimizador
    # weight_decay=0.01 penaliza pesos muy grandes (regularización)
    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = 0.01,
    )

    # Verificar si existe un checkpoint para continuar
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, "best_model.pt")
    epoca_inicio = 0
    if os.path.exists(checkpoint_path):
        print(f"\nCheckpoint encontrado: {checkpoint_path}")
        respuesta = input("  Continuar desde este checkpoint? (s/n): ").strip().lower()
        if respuesta == "s":
            meta = cargar_checkpoint(checkpoint_path, modelo, optimizador)
            epoca_inicio = meta["epoca"]
            print(f"  Reanudando desde época {epoca_inicio + 1}")

    # Loss inicial antes de entrenar (debe ser ~log(89) ≈ 4.49 con pesos aleatorios)
    print("\nEvaluando loss inicial (pesos aleatorios)...")
    loss_inicial = evaluar(modelo, val_loader, device)
    print(f"  Loss inicial: {loss_inicial:.4f}  "
          f"(esperado ~{math.log(VOCAB_SIZE):.2f} = log({VOCAB_SIZE}))")

    # Entrenar
    entrenar(
        modelo       = modelo,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizador  = optimizador,
        device       = device,
        max_epochs   = MAX_EPOCHS,
        epoca_inicio = epoca_inicio,
    )


if __name__ == "__main__":
    main()
