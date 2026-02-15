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
- **TransformerV3** con **Answer Head** (cabeza de regresión numérica auxiliar)
- **10.5M parámetros** — modelo compacto para fines pedagógicos
- **Tokenización BPE** (SentencePiece, vocabulario de 4,000 tokens)
- **Entrenamiento en tres fases**: pre-entrenamiento de encoder → entrenamiento de decoder con cross-attention reinicializada → fine-tuning completo
- **Pipeline completo**: datos → tokenización → entrenamiento → evaluación → interfaz Gradio
- **Interfaz interactiva** con Gradio Blocks para demostración

---

## 🏗️ Arquitectura del Modelo

```
┌──────────────────────────────────────────────────────────────────┐
│                  TRANSFORMER V3 (10.5M params)                   │
│                                                                  │
│  ENCODER (×4 capas)              DECODER (×4 capas)              │
│  ┌──────────────────┐            ┌──────────────────────┐        │
│  │ Input Embedding   │            │ Output Embedding      │        │
│  │ + Pos. Encoding   │            │ + Pos. Encoding       │        │
│  ├──────────────────┤            ├──────────────────────┤        │
│  │ Self-Attention    │──────────▶│ Masked Self-Attention │        │
│  │ (8 heads, d=256)  │           │ Cross-Attention ◀─────┤        │
│  │ Add & LayerNorm   │           │ Add & LayerNorm       │        │
│  │ FFN (1024)        │           │ FFN (1024)            │        │
│  │ Add & LayerNorm   │           │ Add & LayerNorm       │        │
│  └──────────────────┘            └──────────────────────┘        │
│         │                                 │                      │
│    Answer Head                      Linear + Softmax             │
│    (MLP → scalar)                  (vocab_size=4000)             │
└──────────────────────────────────────────────────────────────────┘
```

### Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| `d_model` | 256 |
| `num_heads` | 8 |
| `num_layers` | 4 (encoder) + 4 (decoder) |
| `dff` (feed-forward) | 1024 |
| `dropout_rate` | 0.2 |
| `vocab_size` | 4,000 (BPE / SentencePiece) |
| `max_encoder_len` | 128 tokens |
| `max_decoder_len` | 256 tokens |
| **Total parámetros** | **~10,514,849** |

### Entrenamiento en Tres Fases

El entrenamiento utiliza una estrategia de tres fases para resolver el problema de **colapso de cross-attention** (entropía=1.0):

| Fase | Épocas | Descripción | Componentes entrenados |
|------|--------|-------------|----------------------|
| **Fase 1** | 30 | Pre-entrenamiento del encoder | Encoder + Answer Head (decoder congelado) |
| **Fase 2** | 100 | Entrenamiento del decoder | Decoder + Final Layer (encoder congelado, cross-attention reinicializada) |
| **Fase 3** | 50 | Fine-tuning completo | Todos los parámetros (lr_scale=0.1) |

**Técnicas utilizadas:**
- Optimizador Adam (β₁=0.9, β₂=0.98, ε=1e-9)
- Learning Rate: Warmup (1000 pasos) + inverse sqrt decay
- Loss combinada: seq2seq + answer regression (Huber) + diversity loss
- Decoder token masking (35%) para forzar uso de cross-attention
- Gradient clipping (global norm = 1.0)
- Label smoothing (0.1)
- GPU: NVIDIA RTX 5060 (Blackwell)

---

## 📊 Dataset

Subconjunto curado de **6,881 problemas** con soluciones paso a paso:

| Dominio | Train | Val | Test | Total |
|---------|-------|-----|------|-------|
| Math | ~4,800 | ~420 | ~550 | ~5,770 |
| Physics | ~930 | ~89 | ~93 | ~1,111 |
| **Total** | **5,729** | **509** | **643** | **6,881** |

Derivado de GSM8K, MATH (con soluciones LLM) y problemas de física generados paramétricamente.

---

## 📈 Resultados

| Métrica | Valor |
|---------|-------|
| Token Accuracy (val) | **73.8%** |
| Token Accuracy (test) | **69.9%** |
| Train Accuracy (fase 3) | 64.9% |
| Val Loss | 2.383 |
| Exact Match (Answer:) | **3.0%** (3/100) |
| Exact Match numérico (±0.5) | **3.5%** (3/86) |
| Answer Head MAE | 298.8 |
| Answer Head Exact (±0.5) | 62.2% |

### Cross-Attention — Logro principal

| Capa | Entropía normalizada | Estado |
|------|---------------------|--------|
| Decoder Layer 1 | 0.742 | SELECTIVA |
| Decoder Layer 2 | 0.540 | SELECTIVA |
| Decoder Layer 3 | 0.523 | SELECTIVA |
| Decoder Layer 4 | 0.673 | SELECTIVA |

> **Logro clave**: La cross-attention pasó de colapsada (entropía ≈ 1.0 en v1/v2) a **selectiva** (0.52–0.74), demostrando que el decoder atiende selectivamente al problema de entrada.

### Evolución del proyecto

| Versión | Tokenización | Params | Token Acc | Exact Match | Cross-Attention |
|---------|-------------|--------|-----------|-------------|-----------------|
| v1 | Character (135) | 7.4M | 82.1% | 0% | Colapsada (1.0) |
| v2 | BPE (4000) | 10.5M | ~70% | 0% | Colapsada (1.0) |
| **v3** | **BPE (4000)** | **10.5M** | **73.8%** | **3.0%** | **Selectiva (0.52-0.74)** |

---

## 📁 Estructura del Repositorio

```
transformer_math_physics_tutor/
├── models/                          # Arquitectura Transformer from-scratch
│   ├── transformer.py               #   Base Encoder-Decoder (clase padre)
│   ├── transformer_v3.py            #   TransformerV3 con Answer Head
│   ├── multihead_attention.py       #   Scaled Dot-Product + Multi-Head Attention
│   ├── encoder_layer.py             #   Capa encoder (Self-Attn + FFN)
│   ├── decoder_layer.py             #   Capa decoder (Masked Self-Attn + Cross-Attn + FFN)
│   ├── positional_encoding.py       #   Positional encoding sinusoidal
│   ├── xla_dropout.py               #   Dropout compatible con XLA/Blackwell
│   └── config.py                    #   Configuración del modelo (dataclass)
│
├── data/                            # Pipeline de datos
│   ├── combined_easy.json           #   Dataset curado (6,881 problemas)
│   ├── subword_tokenizer.py         #   Tokenizador BPE (SentencePiece, 4000 tokens)
│   └── dataset_builder.py           #   Constructor de tf.data.Dataset
│
├── training/                        # Loop de entrenamiento
│   ├── trainer.py                   #   TransformerTrainerV3 (GradientTape, 3-phase)
│   ├── losses.py                    #   Loss combinada + diversity loss
│   ├── metrics.py                   #   Exact match + validación simbólica
│   └── scheduler.py                 #   Learning rate: warmup + inverse sqrt + scale
│
├── inference/                       # Generación de respuestas
│   └── generate.py                  #   Generación autoregresiva (greedy, top-k, beam search)
│
├── evaluation/                      # Evaluación del modelo
│   └── evaluate.py                  #   Token accuracy + exact match + cross-attention entropy
│
├── notebooks/                       # Notebooks de demostración
│   ├── 01_exploracion_datos.ipynb   #   Exploración y análisis del dataset
│   ├── 02_entrenamiento.ipynb       #   Notebook de entrenamiento
│   └── 03_demo_profesor.ipynb       #   ⭐ DEMO: Chatbot con interfaz Gradio
│
├── informe final/                   # Informe académico del proyecto
│   └── Informe_MelissaCardona_ChatbotMathPhysics.ipynb
│
├── checkpoints/v3_easy/             # Modelo entrenado (listo para usar)
│   ├── model_weights.weights.h5     #   Pesos del modelo
│   ├── config.json                  #   Configuración
│   ├── sp_tokenizer.model           #   Modelo SentencePiece (BPE)
│   ├── training_history.json        #   Historia (3 fases)
│   └── evaluation_report.json       #   Métricas de evaluación
│
├── run_training.py                  # Script de entrenamiento (3 fases)
├── test_all.py                      # Suite de tests
└── requirements.txt                 # Dependencias Python
```

---

## 🚀 Guía Rápida — Para el Profesor

### ⚠️ El modelo YA viene entrenado. No necesita reentrenar nada.

#### 1. Descargar el proyecto

```bash
git clone https://github.com/MelissaCardona2003/Chat-bot-de-matemacticas-y-f-sica-con-TensorFlow.git
cd Chat-bot-de-matemacticas-y-f-sica-con-TensorFlow
```

#### 2. Crear entorno e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

> Funciona en **CPU**. No necesita GPU.

#### 3. Ejecutar la demo

```bash
jupyter notebook notebooks/03_demo_profesor.ipynb
```

Ejecutar todas las celdas (Shift+Enter). La interfaz Gradio se abrirá automáticamente.

---

## ⚠️ Limitaciones y Trabajo Futuro

1. **3% Exact Match**: Formato correcto pero valores numéricos generalmente incorrectos
2. **Sin mecanismo de copia**: El Transformer estándar no puede copiar tokens directamente del input
3. **Dataset limitado**: 6,881 problemas es pequeño para razonamiento matemático

### ¿Qué SÍ demuestra?

- ✅ Transformer Encoder-Decoder completo from-scratch
- ✅ Pipeline de datos robusto con tokenización BPE
- ✅ Entrenamiento avanzado en tres fases con reinicialización de cross-attention
- ✅ **Cross-attention selectiva** — logro técnico significativo
- ✅ Evaluación rigurosa y honesta
- ✅ Interfaz interactiva Gradio

---

## 📚 Referencias

- Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017
- Cobbe et al., *"Training Verifiers to Solve Math Word Problems"* (GSM8K), 2021
- Hendrycks et al., *"Measuring Mathematical Problem Solving With the MATH Dataset"*, NeurIPS 2021
- Radford et al., *"Language Models are Unsupervised Multitask Learners"* (GPT-2), 2019

---

## 🛠️ Tecnologías

- **TensorFlow 2.x** — Deep learning
- **SentencePiece** — Tokenización BPE
- **Gradio** — Interfaz web
- **NumPy / Matplotlib** — Cálculo y visualización
- **SymPy** — Validación simbólica

---

*Melissa Cardona — 2026*
