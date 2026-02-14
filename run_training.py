"""
run_training.py — Script principal para entrenar el Transformer Tutor.

Ejecuta el pipeline completo:
1. Carga datos de entrenamiento
2. Construye vocabulario (tokenizer)
3. Crea datasets de entrenamiento y validación
4. Instancia el modelo Transformer
5. Entrena con custom training loop
6. Guarda vocabulario, config y pesos
7. Prueba inferencia con problemas de ejemplo

Uso:
    python run_training.py
"""

import os
import sys
import json
import time

# ── GPU / XLA config (ANTES de importar TF) ────────────────
# RTX 5060 (Blackwell, sm_120) requiere XLA con ptxas 12.8
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.8")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

# ── Añadir el directorio padre al path ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import tensorflow as tf

# ── Configurar GPU (memory growth + Blackwell workaround) ───
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] {len(gpus)} GPU(s) detectada(s) — XLA auto-jit activo")

    # Blackwell (sm_120) workaround: TF 2.20 pre-compiled CUDA kernels
    # no soportan sm_120. Forzamos eager tf.cast a CPU; dentro de
    # @tf.function (graph mode) se ejecutan normalmente vía XLA auto-jit.
    _original_cast = tf.cast
    def _blackwell_cast(x, dtype, name=None):
        if tf.executing_eagerly():
            with tf.device('/CPU:0'):
                return _original_cast(x, dtype, name=name)
        return _original_cast(x, dtype, name=name)
    tf.cast = _blackwell_cast
else:
    print("[GPU] No se detectó GPU — entrenando en CPU")

from transformer_math_physics_tutor.data.tokenizer import CharTokenizer
from transformer_math_physics_tutor.data.dataset_builder import create_datasets_from_combined
from transformer_math_physics_tutor.models.config import TransformerConfig
from transformer_math_physics_tutor.models.transformer import Transformer
from transformer_math_physics_tutor.training.train import TransformerTrainer
from transformer_math_physics_tutor.inference.generate import generate_text


def main():
    print("=" * 60)
    print("  TRANSFORMER TUTOR MATEMÁTICO Y FÍSICO")
    print("  Entrenamiento desde cero")
    print("=" * 60)

    # ── 1. Configuración (Dataset combinado Math + Physics) ──────
    DATA_FILE = "combined_math_physics.json"  # Dataset combinado
    MAX_PROBLEM_LEN = 200    # Enunciados ~191 chars promedio
    MAX_SOLUTION_LEN = 300   # Soluciones ~277 chars promedio (physics más largas)
    BATCH_SIZE = 32
    EPOCHS = 100             # Early stopping cortará antes
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    DFF = 1024               # FFN 4× d_model
    DROPOUT = 0.2
    WARMUP_STEPS = 2000
    CHECKPOINT_EVERY = 5
    EARLY_STOPPING_PATIENCE = 10
    LABEL_SMOOTHING = 0.1
    DECODER_MASK_RATE = 0.20  # 20% decoder masking → fuerza cross-attention

    print(f"\n[CONFIG — Math + Physics combinado]")
    print(f"  Data: {DATA_FILE}")
    print(f"  Max problem len: {MAX_PROBLEM_LEN}")
    print(f"  Max solution len: {MAX_SOLUTION_LEN}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  d_model: {D_MODEL}, heads: {NUM_HEADS}, layers: {NUM_LAYERS}, dff: {DFF}")
    print(f"  Dropout: {DROPOUT}, Warmup steps: {WARMUP_STEPS}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Decoder mask rate: {DECODER_MASK_RATE} (fuerza cross-attention)")

    # ── 2. Crear tokenizer y datasets ───────────────────────
    print(f"\n{'='*60}")
    print("PASO 1: Creando tokenizer y datasets...")
    print(f"{'='*60}")

    train_dataset, val_dataset, test_dataset, tokenizer = create_datasets_from_combined(
        data_file=DATA_FILE,
        max_problem_len=MAX_PROBLEM_LEN,
        max_solution_len=MAX_SOLUTION_LEN,
        batch_size=BATCH_SIZE,
        build_vocab=True
    )

    vocab_size = tokenizer.vocab_size
    print(f"\n  Vocabulario construido: {vocab_size} tokens")
    print(f"  Tokens especiales: PAD={tokenizer.pad_token_id}, "
          f"START={tokenizer.start_token_id}, END={tokenizer.end_token_id}, "
          f"UNK={tokenizer.unk_token_id}")

    # ── 3. Guardar vocabulario ──────────────────────────────
    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    tokenizer.save_vocab(vocab_path)
    print(f"  Vocabulario guardado en: {vocab_path}")

    # ── 4. Crear configuración del modelo ───────────────────
    print(f"\n{'='*60}")
    print("PASO 2: Creando modelo Transformer...")
    print(f"{'='*60}")

    config = TransformerConfig(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dff=DFF,
        dropout_rate=DROPOUT,
        max_encoder_len=MAX_PROBLEM_LEN,
        max_decoder_len=MAX_SOLUTION_LEN,
        vocab_size=vocab_size,
        warmup_steps=WARMUP_STEPS,
        label_smoothing=LABEL_SMOOTHING,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_dir="checkpoints"
    )

    # Guardar config
    config_path = os.path.join(checkpoint_dir, "config.json")
    config.save(config_path)
    print(f"  Config guardada en: {config_path}")

    # ── 5. Instanciar modelo (en CPU — Blackwell workaround) ──
    # Variables en CPU; auto-jit=2 maneja GPU durante entrenamiento
    with tf.device('/CPU:0'):
        model = Transformer(config)

        # Build del modelo para ver resumen
        dummy_enc = tf.zeros((1, MAX_PROBLEM_LEN), dtype=tf.int32)
        dummy_dec = tf.zeros((1, MAX_SOLUTION_LEN), dtype=tf.int32)
        _ = model((dummy_enc, dummy_dec), training=False)

    total_params = model.count_params()
    print(f"\n  Modelo creado con {total_params:,} parámetros")
    print(f"  Dispositivo: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")

    # ── 6. Entrenar ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PASO 3: Iniciando entrenamiento...")
    print(f"{'='*60}")

    trainer = TransformerTrainer(model, config,
                                decoder_mask_rate=DECODER_MASK_RATE)

    start_time = time.time()
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        checkpoint_every=CHECKPOINT_EVERY,
        verbose=True,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    total_time = time.time() - start_time

    print(f"\n  Tiempo total de entrenamiento: {total_time:.1f}s "
          f"({total_time/60:.1f} min)")
    print(f"  Loss final: {history['train_loss'][-1]:.4f}")
    print(f"  Accuracy final: {history['train_accuracy'][-1]:.4f}")
    if history['val_loss']:
        print(f"  Val Loss final: {history['val_loss'][-1]:.4f}")
        print(f"  Val Accuracy final: {history['val_accuracy'][-1]:.4f}")

    # ── 7. Guardar pesos del modelo ─────────────────────────
    weights_path = os.path.join(checkpoint_dir, "model_weights.weights.h5")
    trainer.save_model(weights_path)

    # ── 8. Probar inferencia ────────────────────────────────
    print(f"\n{'='*60}")
    print("PASO 4: Probando inferencia...")
    print(f"{'='*60}")

    test_problems = [
        # ── Matemáticas ──
        "Natalia has 3 apples and buys 5 more. How many apples does she have now?",
        "A store sells pencils for $2 each. Tom buys 7 pencils. How much does he spend?",
        "Sarah has 24 cookies. She wants to share them equally among 6 friends. How many cookies does each friend get?",
        "What is 15% of 200?",
        "John has $50. He buys a book for $12 and a pen for $3. How much money does he have left?",
        # ── Física ──
        "A car travels at 60 km/h for 3 hours. What distance does it cover?",
        "What is the net force on a 10 kg object with acceleration 5 m/s^2?",
        "What is the kinetic energy of a 4 kg object moving at 5 m/s?",
        "A 100 W light bulb runs for 5 hours. How much energy does it consume in kWh?",
        "Convert 100°C to Fahrenheit.",
    ]

    print()
    for problem in test_problems:
        with tf.device('/CPU:0'):
            answer = generate_text(
                model, tokenizer, problem,
                max_length=MAX_SOLUTION_LEN,
                temperature=0.3,
                top_k=10,
                repetition_penalty=1.3,
            )
        print(f"  P: {problem}")
        print(f"  R: {answer}")
        print()

    print("=" * 60)
    print("  ¡ENTRENAMIENTO Y PRUEBAS COMPLETADOS!")
    print("=" * 60)
    print(f"\nArchivos guardados en: {checkpoint_dir}/")
    print(f"  - vocab.json")
    print(f"  - config.json")
    print(f"  - model_weights.*")
    print(f"  - training_history.json")


if __name__ == "__main__":
    main()
