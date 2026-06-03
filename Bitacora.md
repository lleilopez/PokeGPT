# Bitácora — PokeGPT

> Mini LLM especializado en Pokémon construido desde cero en Python + PyTorch.  
> Sin modelos preentrenados · Sin embeddings externos · Sin APIs de IA.

---

## Índice

1. [Visión del proyecto](#visión-del-proyecto)
2. [Restricciones absolutas](#restricciones-absolutas)
3. [Arquitectura prevista](#arquitectura-prevista)
4. [Calendario por fases](#calendario-por-fases)
5. [Decisiones técnicas](#decisiones-técnicas)
6. [Registro de sesiones](#registro-de-sesiones)

---

## Visión del proyecto

PokeGPT es un ejercicio de **comprensión profunda** de los LLM modernos.  
El objetivo no es producir el mejor modelo de Pokémon, sino entender *por qué* funciona cada pieza:

- Cómo un tokenizador transforma texto crudo en números.
- Cómo los embeddings dan geometría a esos números.
- Cómo la atención decide qué información importa y cuándo.
- Cómo el entrenamiento autoregresivo enseña al modelo a predecir el siguiente token.
- Cómo la temperatura controla la creatividad en la generación.

Cada componente se implementa a mano, con comentarios en español que explican el *por qué*, no solo el *qué*.

---

## Restricciones absolutas

| Restricción | Motivo |
|---|---|
| Sin modelos preentrenados | Aprender construyendo, no copiando pesos |
| Sin embeddings externos (Word2Vec, GloVe…) | Los embeddings deben surgir del propio entrenamiento |
| Sin APIs de IA (OpenAI, HuggingFace…) | Máxima comprensión, mínima magia negra |
| Solo Python + PyTorch | Stack mínimo y bien documentado |
| Código altamente comentado | Comprensión > rendimiento |
| Comentarios y docstrings en español | Accesibilidad y coherencia del proyecto |
| Nombres de variables y funciones en inglés | Convención estándar de la industria |

---

## Arquitectura prevista

```
PokeGPT/
├── data/
│   ├── raw/           # Texto crudo de la Pokédex y datasets competitivos
│   ├── processed/     # Datasets tokenizados / en formato Q&A
│   └── generated/     # Salidas del pipeline automático (V0.8+)
├── src/
│   ├── tokenizer.py   # Tokenizador por caracteres (V0.1)
│   ├── dataset.py     # Dataset y DataLoader personalizado
│   ├── model.py       # Transformer Decoder completo
│   ├── train.py       # Bucle de entrenamiento
│   ├── generate.py    # Generación de texto con temperatura
│   └── evaluate.py    # Evaluación automática (V0.9+)
├── checkpoints/       # Pesos del modelo guardados por versión
├── logs/              # Curvas de loss y métricas
├── .env               # Variables de entorno locales (no en repo)
├── .env.example       # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
├── Bitacora.md        # Este archivo
└── README.md
```

---

## Calendario por fases

> Inicio del proyecto: semana 1 (junio 2026).  
> Estimación total: ~19 semanas · ~187 horas.

---

### V0.1 — Primer modelo funcional · Semanas 1–3 · ~28h

| Día | Tarea | Estado |
|-----|-------|--------|
| Sem 1 · Lun | Setup: entorno virtual, dependencias PyTorch, estructura de carpetas | [x] |
| Sem 1 · Mar | Cargar texto crudo de la Pokédex simplificada, explorar el dataset | [ ] |
| Sem 1 · Mié | Implementar tokenizador por caracteres: vocab, char2idx, idx2char | [ ] |
| Sem 1 · Jue | Codificar el dataset completo en tensores de índices | [ ] |
| Sem 1 · Vie | Revisar tokenización, imprimir ejemplos, confirmar que funciona | [ ] |
| Sem 1 · Sáb | Implementar capa Embedding desde cero + positional encoding simple | [ ] |
| Sem 1 · Dom | Implementar bloque de atención: Q, K, V, scaled dot-product | [ ] |
| Sem 2 · Lun | Implementar Multi-Head Attention combinando varios bloques | [ ] |
| Sem 2 · Mar | Implementar Feed-Forward + residual connections + layer norm | [ ] |
| Sem 2 · Mié | Ensamblar Transformer Decoder básico (1-2 capas) | [ ] |
| Sem 2 · Jue | Implementar bucle de entrenamiento: forward, loss, backward | [ ] |
| Sem 2 · Vie | Primer entrenamiento: verificar que la loss baja | [ ] |
| Sem 2 · Sáb | Ajustar hiperparámetros: learning rate, batch size, longitud de secuencia | [ ] |
| Sem 2 · Dom | Implementar generación de texto (greedy decoding) | [ ] |
| Sem 3 · Lun | Entrenar modelo completo y evaluar primeras salidas | [ ] |
| Sem 3 · Mar | Corregir problemas evidentes, revisar loss curves | [ ] |
| Sem 3 · Mié | Añadir temperatura a la generación | [ ] |
| Sem 3 · Jue | Documentar el código con comentarios explicativos | [ ] |
| Sem 3 · Vie | Revisión final V0.1: probar prompts, anotar lo aprendido | [ ] |

---

### V0.2 — Pokédex conversacional · Semanas 4–5 · ~18h

| Día | Tarea | Estado |
|-----|-------|--------|
| Sem 4 · Lun | Diseñar formato Q&A: estructura pregunta/respuesta en texto plano | [ ] |
| Sem 4 · Mar | Crear dataset inicial: 50-100 pares sobre tipos y habilidades | [ ] |
| Sem 4 · Mié | Adaptar tokenizador y dataloader al nuevo formato Q&A | [ ] |
| Sem 4 · Jue | Reentrenar con el nuevo dataset | [ ] |
| Sem 4 · Vie | Evaluar respuestas a preguntas directas, anotar errores | [ ] |
| Sem 4 · Sáb | Ampliar dataset: más Pokémon y variantes de pregunta | [ ] |
| Sem 4 · Dom | Experimentar con más capas o heads para mejorar calidad | [ ] |
| Sem 5 · Lun | Añadir token de fin de respuesta para cortar generación | [ ] |
| Sem 5 · Mar | Reentrenar y verificar que las respuestas se cortan bien | [ ] |
| Sem 5 · Mié | Revisión y limpieza del dataset | [ ] |
| Sem 5 · Jue | Documentar cambios respecto a V0.1 | [ ] |
| Sem 5 · Vie | Revisión final V0.2 | [ ] |

---

### V0.3 — Comprensión de tipos · Semana 6 · ~11h

| Día | Tarea | Estado |
|-----|-------|--------|
| Sem 6 · Lun | Modelar tabla de tipos: debilidades, resistencias, inmunidades | [ ] |
| Sem 6 · Mar | Generar Q&A sobre relaciones de tipos (x2, x0.5, x0, x4) | [ ] |
| Sem 6 · Mié | Añadir ejemplos de debilidad doble como Garchomp al Hielo | [ ] |
| Sem 6 · Jue | Reentrenar con el dataset ampliado de tipos | [ ] |
| Sem 6 · Vie | Evaluar respuestas sobre tipos, anotar fallos | [ ] |
| Sem 6 · Sáb | Más variantes de preguntas para cada relación de tipo | [ ] |
| Sem 6 · Dom | Revisión final V0.3 + documentación | [ ] |

---

### V0.4 — Conocimiento competitivo básico · Semanas 7–8 · ~18h

| Día | Tarea | Estado |
|-----|-------|--------|
| Sem 7 · Lun | Definir conceptos: roles, habilidades, objetos frecuentes | [ ] |
| Sem 7 · Mar | Crear ejemplos de roles: sweeper, wall, spinner, pivot… | [ ] |
| Sem 7 · Mié | Crear ejemplos de objetos habituales por arquetipo | [ ] |
| Sem 7 · Jue | Ejemplos de habilidades: Drizzle, Drought, Intimidate… | [ ] |
| Sem 7 · Vie | Reentrenar con el nuevo bloque de datos competitivos | [ ] |
| Sem 7 · Sáb | Evaluar y ajustar ejemplos poco representados | [ ] |
| Sem 7 · Dom | Ampliar con más Pokémon del meta de Showdown | [ ] |
| Sem 8 · Lun | Añadir sinergias simples entre habilidades y clima | [ ] |
| Sem 8 · Mar | Reentrenar y evaluar mejora respecto a V0.3 | [ ] |
| Sem 8 · Mié | Limpiar y balancear el dataset completo acumulado | [ ] |
| Sem 8 · Jue | Documentar cambios | [ ] |
| Sem 8 · Vie | Revisión final V0.4 | [ ] |

---

### V0.5 — Conversación contextual · Semanas 9–11 · ~30h

| Día | Tarea | Estado |
|-----|-------|--------|
| Sem 9 · Lun | Estudiar cómo funciona el context window en LLMs modernos | [ ] |
| Sem 9 · Mar | Rediseñar formato: turnos de conversación multi-vuelta | [ ] |
| Sem 9 · Mié | Implementar concatenado de historial en el input del modelo | [ ] |
| Sem 9 · Jue | Crear ejemplos con referencias pronominales y seguimiento de tema | [ ] |
| Sem 9 · Vie | Primer entrenamiento con el nuevo formato conversacional | [ ] |
| Sem 9 · Sáb | Evaluar si el modelo mantiene referencias entre turnos | [ ] |
| Sem 9 · Dom | Ajustar longitud máxima del contexto, revisar truncado | [ ] |
| Sem 10 · Lun | Ampliar dataset: al menos 80-100 diálogos cortos | [ ] |
| Sem 10 · Mar | Reentrenar y comparar coherencia con V0.4 | [ ] |
| Sem 10 · Mié | Ajustar arquitectura si el modelo no converge bien | [ ] |
| Sem 10 · Jue | Experimentar con 128, 256 y 512 tokens de contexto | [ ] |
| Sem 10 · Vie | Evaluar conversaciones de 3+ turnos con referencias cruzadas | [ ] |
| Sem 10 · Sáb | Iterar sobre casos donde el modelo pierde el contexto | [ ] |
| Sem 10 · Dom | Añadir conversaciones sobre competitivo con contexto | [ ] |
| Sem 11 · Lun | Reentrenamiento final de V0.5 | [ ] |
| Sem 11 · Mar | Benchmarking manual de conversaciones de varios turnos | [ ] |
| Sem 11 · Mié | Documentar cambios arquitectónicos | [ ] |
| Sem 11 · Jue | Limpiar código: refactorizar deuda técnica | [ ] |
| Sem 11 · Vie | Revisión final V0.5 | [ ] |

---

### V0.6 — Dataset competitivo avanzado · Semana 12 · ~11h

| Día | Tarea | Estado |
|-----|-------|--------|
| Sem 12 · Lun | Recopilar sets de Showdown: movimientos, objetos, EVs | [ ] |
| Sem 12 · Mar | Convertir sets a formato Q&A conversacional | [ ] |
| Sem 12 · Mié | Añadir sinergias: lluvia, sol, arena, hail | [ ] |
| Sem 12 · Jue | Reentrenar con dataset competitivo avanzado | [ ] |
| Sem 12 · Vie | Evaluar preguntas sobre sets y sinergias | [ ] |
| Sem 12 · Sáb | Ampliar con más Pokémon de VGC y Smogon OU | [ ] |
| Sem 12 · Dom | Revisión final V0.6 + documentación | [ ] |

---

### V0.7 — Recomendaciones básicas · Semanas 13–14 · ~18h

| Día | Tarea | Estado |
|-----|-------|--------|
| Sem 13 · Lun | Definir tipo de recomendaciones: por rol, clima, cobertura | [ ] |
| Sem 13 · Mar | Crear ejemplos: "tengo equipo de lluvia, ¿qué añado?" | [ ] |
| Sem 13 · Mié | Crear ejemplos de recomendaciones por debilidades no cubiertas | [ ] |
| Sem 13 · Jue | Recomendaciones de movimientos por slot disponible | [ ] |
| Sem 13 · Vie | Reentrenar con ejemplos de recomendación | [ ] |
| Sem 13 · Sáb | Evaluar calidad: ¿son coherentes? ¿son demasiado vagas? | [ ] |
| Sem 13 · Dom | Ampliar con más situaciones típicas de teambuilding | [ ] |
| Sem 14 · Lun | Reentrenar y comparar con V0.6 | [ ] |
| Sem 14 · Mar | Ajustar ejemplos demasiado genéricos | [ ] |
| Sem 14 · Mié | Añadir contra-picks y respuestas a amenazas del meta | [ ] |
| Sem 14 · Jue | Documentar el sistema y sus limitaciones | [ ] |
| Sem 14 · Vie | Revisión final V0.7 | [ ] |

---

### V0.8 — Dataset automático · Semana 15 · ~11h

| Día | Tarea | Estado |
|-----|-------|--------|
| Sem 15 · Lun | Diseñar script de extracción: PokeAPI o datos locales de Showdown | [ ] |
| Sem 15 · Mar | Implementar conversión automática de datos crudos a Q&A | [ ] |
| Sem 15 · Mié | Implementar generación de variantes de preguntas por hecho | [ ] |
| Sem 15 · Jue | Ejecutar el pipeline y revisar calidad del dataset generado | [ ] |
| Sem 15 · Vie | Limpiar ejemplos mal generados, añadir filtros de calidad | [ ] |
| Sem 15 · Sáb | Regenerar dataset con el pipeline y reentrenar | [ ] |
| Sem 15 · Dom | Revisión final V0.8 + documentación del pipeline | [ ] |

---

### V0.9 — PokeGPT Beta · Semanas 16–17 · ~22h

| Día | Tarea | Estado |
|-----|-------|--------|
| Sem 16 · Lun | Auditar dataset acumulado: detectar huecos por categoría | [ ] |
| Sem 16 · Mar | Rellenar huecos: Pokédex, tipos, competitivo, conversación | [ ] |
| Sem 16 · Mié | Reentrenamiento completo con dataset consolidado | [ ] |
| Sem 16 · Jue | Evaluar el modelo en las 6 categorías principales | [ ] |
| Sem 16 · Vie | Anotar fallos sistémicos y categorías más débiles | [ ] |
| Sem 16 · Sáb | Iterar sobre las categorías más débiles | [ ] |
| Sem 16 · Dom | Reentrenar y volver a evaluar | [ ] |
| Sem 17 · Lun | Implementar guardado y carga de checkpoints | [ ] |
| Sem 17 · Mar | Implementar evaluación automática sobre preguntas de test | [ ] |
| Sem 17 · Mié | Crear set de test fijo de 50 preguntas | [ ] |
| Sem 17 · Jue | Ejecutar evaluación y documentar resultados | [ ] |
| Sem 17 · Vie | Revisión final V0.9 | [ ] |

---

### V1.0 — PokeGPT Final · Semanas 18–19 · ~20h

| Día | Tarea | Estado |
|-----|-------|--------|
| Sem 18 · Lun | Revisión arquitectural final: escalar capas, heads o context window | [ ] |
| Sem 18 · Mar | Entrenamiento final con la capacidad del modelo ajustada | [ ] |
| Sem 18 · Mié | Implementar interfaz de consola interactiva | [ ] |
| Sem 18 · Jue | Pulir la generación: evitar repeticiones, mejorar corte | [ ] |
| Sem 18 · Vie | Evaluación completa sobre el set de test | [ ] |
| Sem 18 · Sáb | Sesión de uso real: 2h chateando con PokeGPT, anotar comportamientos | [ ] |
| Sem 18 · Dom | Último ciclo de mejoras basado en la sesión de uso real | [ ] |
| Sem 19 · Lun | Escribir README: arquitectura, dataset, decisiones tomadas | [ ] |
| Sem 19 · Mar | Documentar lecciones aprendidas versión a versión | [ ] |
| Sem 19 · Mié | Limpiar y reorganizar el repositorio | [ ] |
| Sem 19 · Jue | Reentrenamiento final limpio desde cero | [ ] |
| Sem 19 · Vie | Revisión final V1.0: set de test, archivar checkpoint definitivo | [ ] |

---

## Decisiones técnicas

> Se irán añadiendo a medida que se tomen durante el desarrollo.

| Fecha | Decisión | Alternativa descartada | Motivo |
|-------|----------|------------------------|--------|
| 2026-06-03 | Tokenizador por caracteres para V0.1 | BPE / WordPiece | Máxima simplicidad educativa; se puede escalar después |
| 2026-06-03 | Positional encoding sinusoidal simple | Rotary (RoPE) | Más fácil de entender e implementar manualmente |
| 2026-06-03 | Transformer Decoder puro (autoregresivo) | Encoder-Decoder (seq2seq) | Arquitectura de los LLM modernos (GPT-style) |

---

## Registro de sesiones

> Cada sesión de trabajo queda anotada aquí con lo que se hizo, lo que se aprendió y los problemas encontrados.

---

### Sesión 1 — 2026-06-03

**Versión en curso:** Pre-V0.1 — Setup del repositorio  
**Tareas completadas:**
- Creación de `.gitignore`, `.env`, `.env.example`
- Creación de `Bitacora.md` con el calendario completo por fases
- Definición de la arquitectura de carpetas prevista

**Decisiones tomadas:**
- Variables de entorno para hiperparámetros por defecto, facilitando experimentos sin tocar código
- `data/raw/`, `data/processed/`, `data/generated/` separados para tener trazabilidad del origen de cada dato
- Checkpoints ignorados en git (pueden pesar cientos de MB)

**Próxima sesión:** Sem 1 · Mar — Cargar texto crudo de la Pokédex simplificada, explorar el dataset

---

### Sesión 2 — 2026-06-03

**Versión en curso:** V0.1 · Sem 1 · Lun  
**Tareas completadas:**
- Entorno virtual creado con Python 3.12.3 en `.venv/`
- Instaladas dependencias: torch 2.3.1+cpu, numpy 1.26.4, python-dotenv 1.0.1
- Estructura de carpetas creada: `data/raw/`, `data/processed/`, `data/generated/`, `src/`, `checkpoints/`, `logs/`
- Creado `setup.ps1` para reproducir el entorno desde cero
- Creado `verify_setup.py`: todos los checks en verde

**Decisiones tomadas:**
- Python 3.12.3 en lugar de 3.10 (ambos disponibles en el sistema; 3.12 más moderno y soportado por PyTorch)
- PyTorch CPU por defecto; se puede cambiar a CUDA editando `requirements.txt` si se dispone de GPU
- `.gitkeep` en cada carpeta vacía para que git las rastree sin contenido

**Próxima sesión:** Sem 1 · Mar — Cargar texto crudo de la Pokédex simplificada, explorar el dataset

---
