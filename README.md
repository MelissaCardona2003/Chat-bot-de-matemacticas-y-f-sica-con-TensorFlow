# 🎓 Chatbot Tutor de Matemáticas y Física con Transformer

**Modelo Transformer Encoder–Decoder implementado desde cero en TensorFlow**, entrenado para resolver problemas de **matemáticas** (aritmética, álgebra) y **física** (cinemática, dinámica, termodinámica, circuitos) con soluciones paso a paso.

> Proyecto final del curso de profundización en Deep Learning — Carrera de Física  
> Melissa Cardona, 2026

---

## 📋 Descripción

Este proyecto implementa un **Transformer completo (Encoder–Decoder)** desde cero, sin utilizar modelos pre-entrenados como GPT, BERT ni APIs externas. Todo el código de atención, positional encoding, capas encoder/decoder, y el loop de entrenamiento está escrito a mano en TensorFlow/Keras.

El modelo recibe un problema de matemáticas o física como entrada y genera una solución paso a paso con el formato:

```
Step 1: Identify the variables...
Step 2: Apply the formula...
Answer: 42
```

### Características principales

- **Arquitectura from-scratch**: Multi-Head Attention, Positional Encoding sinusoidal, Encoder-Decoder con residual connections
- **7.4M parámetros** — modelo compacto para fines pedagógicos
- **Tokenización a nivel de carácter** (135 tokens: 131 ASCII + 4 especiales)
- **Pipeline completo**: datos → tokenización → entrenamiento → evaluación → interfaz Gradio
- **Interfaz interactiva** con Gradio Blocks para demostración

---

## 🏗️ Arquitectura del Modelo

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER (7.4M params)                │
│                                                             │
│  ENCODER (×4 capas)              DECODER (×4 capas)         │
│  ┌──────────────────┐            ┌──────────────────────┐   │
│  │ Input Embedding   │            │ Output Embedding      │   │
│  │ + Pos. Encoding   │            │ + Pos. Encoding       │   │
│  ├──────────────────┤            ├──────────────────────┤   │
│  │ Self-Attention    │──────────▶│ Masked Self-Attention │   │
│  │ (8 heads, d=256)  │           │ Cross-Attention ◀─────┤   │
│  │ Add & LayerNorm   │           │ Add & LayerNorm       │   │
│  │ FFN (1024)        │           │ FFN (1024)            │   │
│  │ Add & LayerNorm   │           │ Add & LayerNorm       │   │
│  └──────────────────┘            └──────────────────────┘   │
│                                           │                 │
│                                    Linear + Softmax         │
│                                    (vocab_size=135)         │
└─────────────────────────────────────────────────────────────┘
```

### Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| `d_model` | 256 |
| `num_heads` | 8 |
| `num_layers` | 4 (encoder) + 4 (decoder) |
| `dff` (feed-forward) | 1024 |
| `dropout_rate` | 0.2 |
| `vocab_size` | 135 (character-level) |
| `max_encoder_len` | 200 tokens |
| `max_decoder_len` | 300 tokens |
| **Total parámetros** | **7,476,615** |

### Entrenamiento

| Aspecto | Valor |
|---------|-------|
| Optimizador | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| Learning Rate | Warmup (2000 pasos) + inverse sqrt decay |
| Loss | SparseCategoricalCrossentropy + label smoothing (0.1) |
| Batch size | 32 |
| Épocas | 89 (early stopping, patience=10) |
| Regularización | Dropout 0.2, decoder token masking (20%) |
| GPU | NVIDIA RTX 5060 (Blackwell) |

---

## 📊 Datasets

El dataset combinado contiene **12,568 problemas** con soluciones paso a paso:

| Fuente | Dominio | Problemas | Descripción |
|--------|---------|-----------|-------------|
| [GSM8K](https://github.com/openai/grade-school-math) | Math | 8,638 | Aritmética de nivel escolar con razonamiento |
| MATH (LLM-solved) | Math | 1,895 | Álgebra, combinatoria, geometría — soluciones generadas con LLM |
| Physics Templates | Physics | 2,035 | Cinemática, dinámica, termodinámica, circuitos — problemas paramétricos |

**Splits**: Train 10,237 / Val 939 / Test 1,392

---

## 📈 Resultados

| Métrica | Valor |
|---------|-------|
| Token Accuracy (val) | **82.1%** |
| Token Accuracy (test) | **81.2%** |
| Train Accuracy | 73.4% |
| Val Loss | 1.37 |
| Exact Match (Answer:) | 0% (0/100) |

> **Nota importante**: El modelo alcanza ~82% de accuracy a nivel de token (predice bien el siguiente carácter), pero no logra respuestas numéricas correctas. Esto es una limitación inherente de la tokenización a nivel de carácter con un modelo de 7.4M parámetros. Ver la sección de Limitaciones en el notebook de demo y en el informe final.

---

## 📁 Estructura del Repositorio

```
transformer_math_physics_tutor/
├── models/                          # Arquitectura Transformer from-scratch
│   ├── transformer.py               #   Modelo completo Encoder-Decoder
│   ├── multihead_attention.py        #   Scaled Dot-Product + Multi-Head Attention
│   ├── encoder_layer.py             #   Capa encoder (Self-Attn + FFN)
│   ├── decoder_layer.py             #   Capa decoder (Masked Self-Attn + Cross-Attn + FFN)
│   ├── positional_encoding.py       #   Positional encoding sinusoidal
│   ├── xla_dropout.py               #   Dropout compatible con XLA/Blackwell
│   └── config.py                    #   Configuración del modelo (dataclass)
│
├── data/                            # Pipeline de datos
│   ├── combined_math_physics.json   #   Dataset final combinado (12,568 problemas)
│   ├── tokenizer.py                 #   Tokenizador a nivel de carácter
│   ├── dataset_builder.py           #   Constructor de tf.data.Dataset
│   ├── schema.py                    #   Esquema unificado y validación
│   ├── build_combined_dataset.py    #   Script de construcción del dataset final
│   ├── convert_gsm8k.py             #   Descarga y convierte GSM8K
│   ├── convert_math_combined.py     #   Combina GSM8K + MATH_LLM
│   ├── generate_physics_templates.py #  Genera problemas de física paramétricos
│   └── generate_math_solutions_llm.py # Genera soluciones con LLM para MATH
│
├── training/                        # Loop de entrenamiento
│   ├── train.py                     #   TransformerTrainer (GradientTape, checkpointing)
│   ├── losses.py                    #   Loss con label smoothing + masked accuracy
│   ├── metrics.py                   #   Exact match + validación simbólica (SymPy)
│   └── scheduler.py                 #   Learning rate: warmup + inverse sqrt decay
│
├── inference/                       # Generación de respuestas
│   ├── generate.py                  #   Generación autoregresiva (greedy, top-k, beam search)
│   └── chatbot.py                   #   Chatbot interactivo en terminal
│
├── evaluation/                      # Evaluación del modelo
│   └── evaluate_math_physics.py     #   Token accuracy + exact match por dominio
│
├── notebooks/                       # Notebooks de demostración
│   ├── 01_exploracion_datos.ipynb   #   Exploración y análisis del dataset
│   ├── 02_entrenamiento.ipynb       #   Notebook de entrenamiento
│   └── 03_demo_profesor.ipynb       #   ⭐ DEMO: Chatbot con interfaz Gradio
│
├── informe final/                   # Informe académico del proyecto
│   └── Informe_MelissaCardona_ChatbotMathPhysics.ipynb
│
├── checkpoints/                     # Modelo entrenado (listo para usar)
│   ├── best_model.weights.h5        #   Mejores pesos (por val_loss)
│   ├── model_weights.weights.h5     #   Pesos finales
│   ├── config.json                  #   Configuración del modelo
│   ├── vocab.json                   #   Vocabulario del tokenizador
│   ├── training_history.json        #   Historia de entrenamiento (89 épocas)
│   └── evaluation_report.json       #   Métricas de evaluación
│
├── run_training.py                  # Script principal de entrenamiento
├── test_all.py                      # Suite de tests (14 tests)
└── requirements.txt                 # Dependencias Python
```

---

## 🚀 Guía Rápida — Para el Profesor

### ⚠️ El modelo YA viene entrenado. No necesita reentrenar nada.

#### 1. Descargar el proyecto

**Opción A — Git**:
```bash
git clone https://github.com/MelissaCardona2003/Chat-bot-de-matemacticas-y-f-sica-con-TensorFlow.git
cd Chat-bot-de-matemacticas-y-f-sica-con-TensorFlow
```

**Opción B — ZIP**: En GitHub → botón verde **Code** → **Download ZIP** → descomprimir.

#### 2. Crear entorno e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

> Funciona en **CPU**. No necesita GPU para ejecutar la demo.

#### 3. Ejecutar la demo del chatbot

```bash
jupyter notebook notebooks/03_demo_profesor.ipynb
```

Ejecutar **todas las celdas en orden** (Shift+Enter). La interfaz Gradio se abrirá automáticamente con:
- Selector de dominio (math / physics)
- Ejemplos pre-cargados para probar
- Métricas en tiempo real (confianza, perplexity, tiempo)
- Sección de análisis y limitaciones

#### ¿Qué esperar?

- El modelo genera soluciones en formato **"Step 1:... Step 2:... Answer:..."**
- Las respuestas muestran el estilo correcto, pero los **valores numéricos** no son precisos
- Esto es esperado dado el tamaño del modelo (7.4M params vs 117M+ de GPT-2)
- El notebook incluye un análisis detallado de por qué ocurre y qué se necesitaría para mejorar

---

## ⚠️ Limitaciones Conocidas

1. **0% Exact Match**: El modelo produce respuestas con formato correcto pero valores numéricos incorrectos
2. **Tokenización carácter a carácter**: Un problema de 100 palabras → ~500 tokens (vs ~25 con BPE)
3. **Escala del modelo**: 7.4M parámetros (~16x menor que GPT-2 small)
4. **Sin pre-entrenamiento**: Aprende todo desde cero

### ¿Qué SÍ demuestra este proyecto?

- ✅ Implementación correcta de un Transformer Encoder-Decoder completo desde cero
- ✅ Pipeline de datos robusto (descarga, limpieza, validación, schema unificado)
- ✅ Entrenamiento con técnicas modernas (label smoothing, LR scheduling, early stopping)
- ✅ Evaluación honesta y rigurosa con métricas apropiadas
- ✅ Despliegue con interfaz interactiva profesional (Gradio)

---

## 📚 Referencias

- Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017
- Cobbe et al., *"Training Verifiers to Solve Math Word Problems"* (GSM8K), 2021
- Hendrycks et al., *"Measuring Mathematical Problem Solving With the MATH Dataset"*, NeurIPS 2021
- Radford et al., *"Language Models are Unsupervised Multitask Learners"* (GPT-2), 2019

---

## 🛠️ Tecnologías

- **TensorFlow 2.x** — Framework de deep learning
- **Gradio** — Interfaz web interactiva
- **NumPy / Matplotlib** — Cálculo numérico y visualización
- **SymPy** — Validación simbólica de respuestas
- **Datasets (HuggingFace)** — Descarga de datasets

---

*Melissa Cardona — 2026*
