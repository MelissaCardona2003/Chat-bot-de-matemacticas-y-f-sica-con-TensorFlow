"""
run_training.py — Two-Phase Training for Transformer Math/Physics Tutor.

Solves cross-attention collapse (entropy=1.0) via two phases:

Phase 1 — Encoder Pre-training (30 epochs):
  - Freeze decoder + final_layer
  - Only encoder + answer_head get gradients
  - answer_loss forces encoder to extract numerical information
  - No decoder masking, no diversity loss

Phase 2 — Full Training with Fresh Cross-Attention (100 epochs):
  - REINITIALIZE cross-attention Q/K/V weights (breaks symmetric collapse)
  - Unfreeze all layers
  - seq2seq + answer_loss + diversity_loss
  - 35% decoder masking

Key insight: v2 transfer weights have collapsed cross-attention (entropy=1.0).
Phase 1 pre-trains encoder to produce MEANINGFUL representations.
Phase 2 reinitializes cross-attention so it discovers those representations.

Uso:
    python run_training.py
"""

import os
import sys
import json
import time
import shutil

# ── GPU / XLA config (ANTES de importar TF) ────────────────
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.8")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

# ── Path ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import tensorflow as tf
import numpy as np

# ── GPU setup + Blackwell workaround ────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] {len(gpus)} GPU(s) detectada(s) — XLA auto-jit activo")

    _original_cast = tf.cast
    def _blackwell_cast(x, dtype, name=None):
        if tf.executing_eagerly():
            with tf.device('/CPU:0'):
                return _original_cast(x, dtype, name=name)
        return _original_cast(x, dtype, name=name)
    tf.cast = _blackwell_cast
else:
    print("[GPU] No se detectó GPU — entrenando en CPU")

from transformer_math_physics_tutor.data.subword_tokenizer import SubwordTokenizer
from transformer_math_physics_tutor.data.dataset_builder import create_datasets_v3_easy
from transformer_math_physics_tutor.models.config import TransformerConfig
from transformer_math_physics_tutor.models.transformer import Transformer
from transformer_math_physics_tutor.models.transformer_v3 import TransformerV3
from transformer_math_physics_tutor.training.trainer import TransformerTrainerV3
from transformer_math_physics_tutor.inference.generate import generate_text


def reinitialize_cross_attention(model):
    """
    Reinitializa los pesos de cross-attention (mha2) en cada decoder layer.

    Esto rompe el colapso simétrico de la cross-attention heredado de v2.
    Las Q/K/V projections se reinicializan con Glorot uniform.
    El LayerNorm de cross-attention se resetea a gamma=1, beta=0.

    Args:
        model: TransformerV3 con decoder.dec_layers
    """
    n_reinitialized = 0
    with tf.device('/CPU:0'):
        for i, layer in enumerate(model.decoder.dec_layers):
            # Reinitializar MultiHeadAttention de cross-attention (mha2)
            for sublayer in [layer.mha2.wq, layer.mha2.wk, layer.mha2.wv, layer.mha2.dense]:
                kernel = sublayer.kernel
                bias = sublayer.bias
                # Glorot uniform
                fan_in, fan_out = int(kernel.shape[0]), int(kernel.shape[1])
                limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
                new_kernel = np.random.uniform(-limit, limit, size=kernel.shape).astype(np.float32)
                kernel.assign(new_kernel)
                bias.assign(np.zeros(bias.shape, dtype=np.float32))

            # Reinitializar LayerNorm de cross-attention
            layer.layernorm2.gamma.assign(np.ones(layer.layernorm2.gamma.shape, dtype=np.float32))
            layer.layernorm2.beta.assign(np.zeros(layer.layernorm2.beta.shape, dtype=np.float32))
            n_reinitialized += 1

    print(f"  ✅ Cross-attention reinicializada en {n_reinitialized} decoder layers")
    print(f"     (Q/K/V projections + LayerNorm → Glorot uniform + zeros)")


def main():
    print("=" * 60)
    print("  TRANSFORMER TUTOR — Two-Phase Training")
    print("  Phase 1: Encoder pre-training (answer head)")
    print("  Phase 2: Full training + fresh cross-attention")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════
    #  CONFIGURACIÓN
    # ══════════════════════════════════════════════════════════
    DATA_FILE = "combined_easy.json"
    TOKENIZER_MODEL = "checkpoints/v2_subword/sp_tokenizer.model"
    V2_WEIGHTS = "checkpoints/v2_subword/best_model.weights.h5"

    MAX_PROBLEM_LEN = 128
    MAX_SOLUTION_LEN = 256
    BATCH_SIZE = 32
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    DFF = 1024
    DROPOUT = 0.2
    LABEL_SMOOTHING = 0.1
    ANSWER_SCALE = 1000.0
    CHECKPOINT_DIR = "checkpoints/v3_easy"

    # ── Phase 1 config ──
    P1_EPOCHS = 30
    P1_WARMUP_STEPS = 500
    P1_ANSWER_WEIGHT = 10.0      # Dominante — fuerza encoder learning
    P1_DECODER_MASK_RATE = 0.0   # Sin masking (decoder congelado)
    P1_DIVERSITY_WEIGHT = 0.0    # Sin diversity (cross-attention congelada)

    # ── Phase 2 config ──
    P2_EPOCHS = 100
    P2_WARMUP_STEPS = 1000
    P2_ANSWER_WEIGHT = 5.0
    P2_DECODER_MASK_RATE = 0.35
    P2_DIVERSITY_WEIGHT = 10.0
    P2_PATIENCE = 0              # Sin early stopping — train all epochs
    P2_CHECKPOINT_EVERY = 5

    print(f"\n[CONFIG]")
    print(f"  Data: {DATA_FILE}")
    print(f"  Architecture: d={D_MODEL}, h={NUM_HEADS}, L={NUM_LAYERS}, dff={DFF}")
    print(f"  Phase 1: {P1_EPOCHS} epochs, answer_weight={P1_ANSWER_WEIGHT}, decoder FROZEN")
    print(f"  Phase 2: {P2_EPOCHS} epochs, answer_weight={P2_ANSWER_WEIGHT}, "
          f"diversity={P2_DIVERSITY_WEIGHT}, mask={P2_DECODER_MASK_RATE}")

    # ══════════════════════════════════════════════════════════
    #  PASO 1: Cargar datos
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PASO 1: Cargando datos y tokenizer...")
    print(f"{'='*60}")

    train_dataset, val_dataset, test_dataset, tokenizer = create_datasets_v3_easy(
        data_file=DATA_FILE,
        tokenizer_model=TOKENIZER_MODEL,
        max_problem_len=MAX_PROBLEM_LEN,
        max_solution_len=MAX_SOLUTION_LEN,
        batch_size=BATCH_SIZE,
    )

    vocab_size = tokenizer.vocab_size
    print(f"\n  Vocabulario: {vocab_size} tokens (BPE)")

    # ══════════════════════════════════════════════════════════
    #  PASO 2: Crear modelo + transfer v2
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PASO 2: Creando modelo + transfer learning v2...")
    print(f"{'='*60}")

    checkpoint_dir = os.path.join(SCRIPT_DIR, CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Config for Phase 1 (will create new config for Phase 2)
    config_p1 = TransformerConfig(
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        dff=DFF, dropout_rate=DROPOUT,
        max_encoder_len=MAX_PROBLEM_LEN, max_decoder_len=MAX_SOLUTION_LEN,
        vocab_size=vocab_size, warmup_steps=P1_WARMUP_STEPS,
        label_smoothing=LABEL_SMOOTHING, epochs=P1_EPOCHS,
        batch_size=BATCH_SIZE, checkpoint_dir=CHECKPOINT_DIR + "/phase1",
    )

    with tf.device('/CPU:0'):
        dummy_enc = tf.zeros((1, MAX_PROBLEM_LEN), dtype=tf.int32)
        dummy_dec = tf.zeros((1, MAX_SOLUTION_LEN), dtype=tf.int32)

        # Modelo v2 temporal (para copiar pesos)
        v2_model = Transformer(config_p1)
        _ = v2_model((dummy_enc, dummy_dec), training=False)

        # Modelo v3
        v3_model = TransformerV3(config_p1, answer_scale=ANSWER_SCALE)
        _ = v3_model((dummy_enc, dummy_dec), training=False)

    # Cargar pesos v2
    v2_weights_path = os.path.join(SCRIPT_DIR, V2_WEIGHTS)
    if os.path.exists(v2_weights_path):
        v2_model.load_weights(v2_weights_path)
        print(f"  ✅ Pesos v2 cargados desde {v2_weights_path}")

        v3_model.encoder.set_weights(v2_model.encoder.get_weights())
        v3_model.decoder.set_weights(v2_model.decoder.get_weights())
        v3_model.final_layer.set_weights(v2_model.final_layer.get_weights())
        print(f"  ✅ Pesos copiados: encoder, decoder, final_layer")
        print(f"  📝 Answer head inicializado aleatoriamente")
        del v2_model
    else:
        print(f"  ⚠️ No se encontró {v2_weights_path} — entrenando desde cero")
        del v2_model

    total_params = v3_model.count_params()
    print(f"\n  Modelo: {total_params:,} parámetros totales")

    # ══════════════════════════════════════════════════════════
    #  FASE 1: Pre-training del Encoder (solo answer head)
    # ══════════════════════════════════════════════════════════
    phase1_dir = os.path.join(checkpoint_dir, "phase1")
    p1_weights_path = os.path.join(phase1_dir, "encoder_pretrained.weights.h5")

    if os.path.exists(p1_weights_path):
        # ── Phase 1 ya completada — cargar pesos ──
        print(f"\n{'='*60}")
        print("FASE 1: Cargando pesos pre-entrenados del encoder...")
        print(f"  (Phase 1 ya completada previamente)")
        print(f"{'='*60}")
        v3_model.load_weights(p1_weights_path)
        print(f"  ✅ Pesos Phase 1 cargados desde {p1_weights_path}")
        time_p1 = 0.0
        history_p1 = {"train_answer_mae": [0.0], "val_answer_mae": [0.0]}
    else:
        print(f"\n{'='*60}")
        print("FASE 1: Pre-training del Encoder")
        print("  Decoder CONGELADO — solo encoder + answer_head reciben gradientes")
        print("  Objetivo: encoder aprende a extraer información numérica")
        print(f"{'='*60}")

        # Congelar decoder y final_layer
        v3_model.decoder.trainable = False
        v3_model.final_layer.trainable = False

        trainable_p1 = sum(v.numpy().size for v in v3_model.trainable_variables)
        frozen_p1 = total_params - trainable_p1
        print(f"  Trainable: {trainable_p1:,} params")
        print(f"  Frozen:    {frozen_p1:,} params")

        os.makedirs(phase1_dir, exist_ok=True)

        trainer_p1 = TransformerTrainerV3(
            model=v3_model,
            config=config_p1,
            answer_weight=P1_ANSWER_WEIGHT,
            answer_scale=ANSWER_SCALE,
            decoder_mask_rate=P1_DECODER_MASK_RATE,
            attn_diversity_weight=P1_DIVERSITY_WEIGHT,
        )

        start_p1 = time.time()
        history_p1 = trainer_p1.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=P1_EPOCHS,
            checkpoint_every=10,
            verbose=True,
            early_stopping_patience=0,  # Sin early stopping — correr todas las épocas
        )
        time_p1 = time.time() - start_p1

        print(f"\n  ── Fase 1 completada en {time_p1:.1f}s ({time_p1/60:.1f} min) ──")
        print(f"  Answer MAE final: {history_p1['train_answer_mae'][-1]:.1f}")
        if history_p1['val_answer_mae']:
            print(f"  Val Answer MAE final: {history_p1['val_answer_mae'][-1]:.1f}")
            print(f"  Val Answer MAE inicio: {history_p1['val_answer_mae'][0]:.1f}")

        # Guardar pesos Phase 1
        v3_model.save_weights(p1_weights_path)
        print(f"  Pesos Phase 1 guardados: {p1_weights_path}")

    # ══════════════════════════════════════════════════════════
    #  FASE 2: Decoder Training + Fresh Cross-Attention
    #  Encoder CONGELADO — preserva representaciones de Phase 1
    # ══════════════════════════════════════════════════════════
    p2_weights_path = os.path.join(checkpoint_dir, "phase2_model.weights.h5")

    if os.path.exists(p2_weights_path):
        print(f"\n{'='*60}")
        print("FASE 2: Cargando pesos de Phase 2 (cross-attention selectiva)...")
        print(f"  (Phase 2 ya completada previamente)")
        print(f"{'='*60}")
        v3_model.load_weights(p2_weights_path)
        print(f"  ✅ Pesos Phase 2 cargados desde {p2_weights_path}")
        time_p2 = 0.0
        history_p2 = {"train_loss": [0], "train_accuracy": [0], "train_answer_mae": [0],
                       "val_loss": [0], "val_accuracy": [0], "val_answer_mae": [0]}
    else:
        print(f"\n{'='*60}")
        print("FASE 2: Decoder Training con Cross-Attention Reinicializada")
        print("  Cross-attention Q/K/V → Glorot uniform (rompe colapso simétrico)")
        print("  Encoder CONGELADO — preserva representaciones pre-entrenadas")
        print("  Decoder descongelado — aprende a USAR el encoder")
        print(f"{'='*60}")

        # Descongelar decoder/final_layer, MANTENER encoder congelado
        v3_model.encoder.trainable = False      # CLAVE: preservar Phase 1
        v3_model.decoder.trainable = True
        v3_model.final_layer.trainable = True

        # CLAVE: Reinicializar cross-attention (rompe el colapso de v2)
        reinitialize_cross_attention(v3_model)

        trainable_p2 = sum(v.numpy().size for v in v3_model.trainable_variables)
        frozen_p2 = total_params - trainable_p2
        print(f"  Trainable: {trainable_p2:,} params (decoder + cross-attn + answer head)")
        print(f"  Frozen:    {frozen_p2:,} params (encoder)")

        # Config para Phase 2 (fresh optimizer, new warmup)
        config_p2 = TransformerConfig(
            d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            dff=DFF, dropout_rate=DROPOUT,
            max_encoder_len=MAX_PROBLEM_LEN, max_decoder_len=MAX_SOLUTION_LEN,
            vocab_size=vocab_size, warmup_steps=P2_WARMUP_STEPS,
            label_smoothing=LABEL_SMOOTHING, epochs=P2_EPOCHS,
            batch_size=BATCH_SIZE, checkpoint_dir=CHECKPOINT_DIR,
        )

        # Guardar config final
        config_path = os.path.join(checkpoint_dir, "config.json")
        config_p2.save(config_path)

        # Limpiar checkpoints de fases anteriores
        for f in os.listdir(checkpoint_dir):
            if f.startswith("ckpt-") or f == "checkpoint":
                os.remove(os.path.join(checkpoint_dir, f))

        trainer_p2 = TransformerTrainerV3(
            model=v3_model,
            config=config_p2,
            answer_weight=P2_ANSWER_WEIGHT,
            answer_scale=ANSWER_SCALE,
            decoder_mask_rate=P2_DECODER_MASK_RATE,
            attn_diversity_weight=P2_DIVERSITY_WEIGHT,
        )

        start_p2 = time.time()
        history_p2 = trainer_p2.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=P2_EPOCHS,
            checkpoint_every=P2_CHECKPOINT_EVERY,
            verbose=True,
            early_stopping_patience=P2_PATIENCE,
        )
        time_p2 = time.time() - start_p2

        print(f"\n  ── Fase 2 completada en {time_p2:.1f}s ({time_p2/60:.1f} min) ──")
        print(f"  Loss final: {history_p2['train_loss'][-1]:.4f}")
        print(f"  Accuracy final: {history_p2['train_accuracy'][-1]:.4f}")
        print(f"  Answer MAE final: {history_p2['train_answer_mae'][-1]:.1f}")
        if history_p2['val_loss']:
            print(f"  Val Loss final: {history_p2['val_loss'][-1]:.4f}")
            print(f"  Val Accuracy final: {history_p2['val_accuracy'][-1]:.4f}")
            print(f"  Val MAE final: {history_p2['val_answer_mae'][-1]:.1f}")

        # Guardar pesos Phase 2
        v3_model.save_weights(p2_weights_path)
        print(f"  Pesos Phase 2 guardados: {p2_weights_path}")

    # ══════════════════════════════════════════════════════════
    #  FASE 3: Fine-tuning con encoder descongelado
    #  Cross-attention ya selectiva → gradiente enseña al encoder
    #  a codificar identidad token-level (números específicos)
    # ══════════════════════════════════════════════════════════
    P3_EPOCHS = 50
    P3_WARMUP_STEPS = 2000       # Warmup lento → learning rate baja
    P3_LR_SCALE = 0.1            # Peak LR ~0.00014 (fine-tuning, no destruir pesos)
    P3_ANSWER_WEIGHT = 5.0
    P3_DIVERSITY_WEIGHT = 2.0    # Cross-attention ya selectiva, menos push
    P3_DECODER_MASK_RATE = 0.35

    print(f"\n{'='*60}")
    print("FASE 3: Fine-tuning con Encoder Descongelado")
    print("  Cross-attention selectiva → gradiente llega al encoder")
    print("  Encoder aprende identidad token-level (números específicos)")
    print("  Learning rate baja para no destruir patrones aprendidos")
    print(f"  LR scale: {P3_LR_SCALE} → peak LR ~{0.0625 * 0.02236 * P3_LR_SCALE:.6f}")
    print(f"{'='*60}")

    # Descongelar encoder
    v3_model.encoder.trainable = True

    trainable_p3 = sum(v.numpy().size for v in v3_model.trainable_variables)
    print(f"  Trainable: {trainable_p3:,} params (TODOS)")

    config_p3 = TransformerConfig(
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        dff=DFF, dropout_rate=DROPOUT,
        max_encoder_len=MAX_PROBLEM_LEN, max_decoder_len=MAX_SOLUTION_LEN,
        vocab_size=vocab_size, warmup_steps=P3_WARMUP_STEPS,
        lr_scale=P3_LR_SCALE,
        label_smoothing=LABEL_SMOOTHING, epochs=P3_EPOCHS,
        batch_size=BATCH_SIZE, checkpoint_dir=CHECKPOINT_DIR,
    )

    # Limpiar checkpoints de Phase 2
    for f in os.listdir(checkpoint_dir):
        if f.startswith("ckpt-") or f == "checkpoint":
            os.remove(os.path.join(checkpoint_dir, f))

    trainer_p3 = TransformerTrainerV3(
        model=v3_model,
        config=config_p3,
        answer_weight=P3_ANSWER_WEIGHT,
        answer_scale=ANSWER_SCALE,
        decoder_mask_rate=P3_DECODER_MASK_RATE,
        attn_diversity_weight=P3_DIVERSITY_WEIGHT,
    )

    start_p3 = time.time()
    history_p3 = trainer_p3.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=P3_EPOCHS,
        checkpoint_every=P2_CHECKPOINT_EVERY,
        verbose=True,
        early_stopping_patience=0,  # Sin early stopping — train all
    )
    time_p3 = time.time() - start_p3

    print(f"\n  ── Fase 3 completada en {time_p3:.1f}s ({time_p3/60:.1f} min) ──")
    print(f"  Loss final: {history_p3['train_loss'][-1]:.4f}")
    print(f"  Accuracy final: {history_p3['train_accuracy'][-1]:.4f}")
    print(f"  Answer MAE final: {history_p3['train_answer_mae'][-1]:.1f}")
    if history_p3['val_loss']:
        print(f"  Val Loss final: {history_p3['val_loss'][-1]:.4f}")
        print(f"  Val Accuracy final: {history_p3['val_accuracy'][-1]:.4f}")
        print(f"  Val MAE final: {history_p3['val_answer_mae'][-1]:.1f}")

    total_time = time_p1 + time_p2 + time_p3
    print(f"\n  Tiempo total (3 fases): {total_time:.1f}s ({total_time/60:.1f} min)")

    # ══════════════════════════════════════════════════════════
    #  GUARDAR MODELO FINAL
    # ══════════════════════════════════════════════════════════
    weights_path = os.path.join(checkpoint_dir, "model_weights.weights.h5")
    v3_model.save_weights(weights_path)
    print(f"Pesos guardados en {weights_path}")

    # Copiar tokenizer
    src_tok = os.path.join(SCRIPT_DIR, TOKENIZER_MODEL)
    dst_tok = os.path.join(checkpoint_dir, "sp_tokenizer.model")
    if os.path.exists(src_tok):
        shutil.copy2(src_tok, dst_tok)
        print(f"  Tokenizer copiado a {dst_tok}")

    # Guardar historial combinado
    combined_history = {
        "phase1": history_p1,
        "phase2": history_p2,
        "phase3": history_p3,
        "phase1_time_s": time_p1,
        "phase2_time_s": time_p2,
        "phase3_time_s": time_p3,
    }
    hist_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(combined_history, f, indent=2)

    # ══════════════════════════════════════════════════════════
    #  PROBAR INFERENCIA
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PASO FINAL: Probando inferencia...")
    print(f"{'='*60}")

    test_problems = [
        "Natalia has 3 apples and buys 5 more. How many apples does she have now?",
        "A store sells pencils for $2 each. Tom buys 7 pencils. How much does he spend?",
        "Sarah has 24 cookies. She wants to share them equally among 6 friends. How many cookies does each friend get?",
        "What is 15% of 200?",
        "A car travels at 60 km/h for 3 hours. What distance does it cover?",
        "What is the net force on a 10 kg object with acceleration 5 m/s^2?",
        "What is the kinetic energy of a 4 kg object moving at 5 m/s?",
        "A wave has frequency 100 Hz and wavelength 0.1 m. What is the wave speed?",
    ]

    print()
    for problem in test_problems:
        with tf.device('/CPU:0'):
            answer = generate_text(
                v3_model, tokenizer, problem,
                max_length=MAX_SOLUTION_LEN,
                temperature=0.3,
                top_k=10,
                repetition_penalty=1.3,
            )
        print(f"  P: {problem}")
        print(f"  R: {answer[:200]}")
        print()

    print("=" * 60)
    print("  ¡ENTRENAMIENTO v3_easy COMPLETADO!")
    print("=" * 60)
    print(f"\nArchivos en: {checkpoint_dir}/")
    print(f"  - config.json")
    print(f"  - sp_tokenizer.model")
    print(f"  - model_weights.weights.h5")
    print(f"  - best_model.weights.h5")
    print(f"  - training_history.json")


if __name__ == "__main__":
    main()
