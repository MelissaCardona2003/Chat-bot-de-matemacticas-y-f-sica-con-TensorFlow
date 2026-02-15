"""
run_training_v4.py — Three-Phase Training for TransformerV4 (Pointer-Generator).

Same three-phase strategy as v3, but with copy mechanism:

Phase 1 — Encoder Pre-training (30 epochs):
  - Freeze decoder + final_layer + p_gen_gate
  - Only encoder + answer_head get gradients
  - Builds meaningful encoder representations for copy mechanism

Phase 2 — Decoder Training + Fresh Cross-Attention (100 epochs):
  - REINITIALIZE cross-attention Q/K/V (break symmetric collapse)
  - Encoder FROZEN — preserve Phase 1 representations
  - Decoder + p_gen learn to use cross-attention AND copy mechanism
  - 35% decoder masking → forces cross-attention dependency
  - Cross-attention diversity loss → prevents attention collapse

Phase 3 — Fine-tuning (50 epochs):
  - ALL layers unfrozen (including encoder)
  - Low learning rate (0.1x) to preserve patterns
  - Copy mechanism fully integrated — encoder + decoder co-adapt

Key insight: The copy mechanism builds on top of selective cross-attention.
Phase 2's fresh cross-attention gives p_gen clean attention weights to
decide WHEN to copy. Phase 3 lets the encoder learn to produce
representations that are easy to copy FROM.

Uso:
    python run_training_v4.py
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
from transformer_math_physics_tutor.models.transformer_v3 import TransformerV3
from transformer_math_physics_tutor.models.transformer_v4 import TransformerV4
from transformer_math_physics_tutor.training.trainer_v4 import TransformerTrainerV4
from transformer_math_physics_tutor.inference.generate_v4 import generate_text_v4


def reinitialize_cross_attention(model):
    """
    Reinitializa los pesos de cross-attention (mha2) en cada decoder layer.

    Rompe el colapso simétrico de la cross-attention.
    Q/K/V projections se reinicializan con Glorot uniform.
    LayerNorm de cross-attention se resetea a gamma=1, beta=0.

    Args:
        model: TransformerV4 con decoder.dec_layers
    """
    n_reinitialized = 0
    with tf.device('/CPU:0'):
        for i, layer in enumerate(model.decoder.dec_layers):
            for sublayer in [layer.mha2.wq, layer.mha2.wk, layer.mha2.wv, layer.mha2.dense]:
                kernel = sublayer.kernel
                bias = sublayer.bias
                fan_in, fan_out = int(kernel.shape[0]), int(kernel.shape[1])
                limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
                new_kernel = np.random.uniform(-limit, limit, size=kernel.shape).astype(np.float32)
                kernel.assign(new_kernel)
                bias.assign(np.zeros(bias.shape, dtype=np.float32))

            layer.layernorm2.gamma.assign(np.ones(layer.layernorm2.gamma.shape, dtype=np.float32))
            layer.layernorm2.beta.assign(np.zeros(layer.layernorm2.beta.shape, dtype=np.float32))
            n_reinitialized += 1

    print(f"  ✅ Cross-attention reinicializada en {n_reinitialized} decoder layers")


def main():
    print("=" * 60)
    print("  TRANSFORMER V4 — Pointer-Generator Training")
    print("  Phase 1: Encoder pre-training (answer head)")
    print("  Phase 2: Decoder + copy mechanism training")
    print("  Phase 3: Fine-tuning all layers")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════
    #  CONFIGURACIÓN
    # ══════════════════════════════════════════════════════════
    DATA_FILE = "combined_easy.json"
    TOKENIZER_MODEL = "checkpoints/v2_subword/sp_tokenizer.model"
    V3_WEIGHTS = "checkpoints/v3_easy/model_weights.weights.h5"

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
    NUM_COPY_LAYERS = 2  # Últimas 2 capas de cross-attention para copy
    CHECKPOINT_DIR = "checkpoints/v4_copy"

    # ── Phase 1 config ──
    P1_EPOCHS = 30
    P1_WARMUP_STEPS = 500
    P1_ANSWER_WEIGHT = 10.0
    P1_DECODER_MASK_RATE = 0.0
    P1_DIVERSITY_WEIGHT = 0.0

    # ── Phase 2 config ──
    P2_EPOCHS = 100
    P2_WARMUP_STEPS = 1000
    P2_ANSWER_WEIGHT = 5.0
    P2_DECODER_MASK_RATE = 0.35
    P2_DIVERSITY_WEIGHT = 10.0
    P2_PATIENCE = 0
    P2_CHECKPOINT_EVERY = 5

    # ── Phase 3 config ──
    P3_EPOCHS = 50
    P3_WARMUP_STEPS = 2000
    P3_LR_SCALE = 0.1
    P3_ANSWER_WEIGHT = 5.0
    P3_DIVERSITY_WEIGHT = 2.0
    P3_DECODER_MASK_RATE = 0.35

    print(f"\n[CONFIG]")
    print(f"  Data: {DATA_FILE}")
    print(f"  Architecture: d={D_MODEL}, h={NUM_HEADS}, L={NUM_LAYERS}, dff={DFF}")
    print(f"  Copy mechanism: num_copy_layers={NUM_COPY_LAYERS}")
    print(f"  Phase 1: {P1_EPOCHS} epochs, encoder only, answer_weight={P1_ANSWER_WEIGHT}")
    print(f"  Phase 2: {P2_EPOCHS} epochs, decoder+copy, diversity={P2_DIVERSITY_WEIGHT}")
    print(f"  Phase 3: {P3_EPOCHS} epochs, fine-tune all, lr_scale={P3_LR_SCALE}")

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
    #  PASO 2: Crear modelo V4 + transfer learning desde V3
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PASO 2: Creando TransformerV4 + transfer learning desde v3...")
    print(f"{'='*60}")

    checkpoint_dir = os.path.join(SCRIPT_DIR, CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)

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

        # Crear modelo V4
        v4_model = TransformerV4(config_p1, answer_scale=ANSWER_SCALE,
                                 num_copy_layers=NUM_COPY_LAYERS)
        _ = v4_model((dummy_enc, dummy_dec), training=False)

    # Transfer learning desde V3
    v3_weights_path = os.path.join(SCRIPT_DIR, V3_WEIGHTS)
    if os.path.exists(v3_weights_path):
        # Crear V3 temporal para copiar pesos
        with tf.device('/CPU:0'):
            v3_temp = TransformerV3(config_p1, answer_scale=ANSWER_SCALE)
            _ = v3_temp((dummy_enc, dummy_dec), training=False)
        v3_temp.load_weights(v3_weights_path)
        print(f"  ✅ Pesos v3 cargados desde {v3_weights_path}")

        # Copiar pesos compartidos: encoder, decoder, final_layer, answer_head
        v4_model.encoder.set_weights(v3_temp.encoder.get_weights())
        v4_model.decoder.set_weights(v3_temp.decoder.get_weights())
        v4_model.final_layer.set_weights(v3_temp.final_layer.get_weights())
        v4_model.answer_dense1.set_weights(v3_temp.answer_dense1.get_weights())
        v4_model.answer_dropout.set_weights(v3_temp.answer_dropout.get_weights())
        v4_model.answer_dense2.set_weights(v3_temp.answer_dense2.get_weights())
        print(f"  ✅ Pesos copiados: encoder, decoder, final_layer, answer_head")
        print(f"  📝 p_gen gate inicializado aleatoriamente (bias=1.0 → p_gen≈0.73)")
        del v3_temp
    else:
        print(f"  ⚠️ No se encontró {v3_weights_path} — entrenando desde cero")

    total_params = v4_model.count_params()
    # Count only copy-specific params
    copy_params = sum(v.numpy().size for v in v4_model.p_gen_linear.trainable_variables)
    print(f"\n  Modelo V4: {total_params:,} parámetros totales")
    print(f"  Params nuevos (p_gen gate): {copy_params:,}")
    print(f"  Params heredados de v3: {total_params - copy_params:,}")

    # ══════════════════════════════════════════════════════════
    #  FASE 1: Pre-training del Encoder (solo answer head)
    # ══════════════════════════════════════════════════════════
    phase1_dir = os.path.join(checkpoint_dir, "phase1")
    p1_weights_path = os.path.join(phase1_dir, "encoder_pretrained.weights.h5")

    if os.path.exists(p1_weights_path):
        print(f"\n{'='*60}")
        print("FASE 1: Cargando pesos pre-entrenados del encoder...")
        print(f"  (Phase 1 ya completada previamente)")
        print(f"{'='*60}")
        v4_model.load_weights(p1_weights_path)
        print(f"  ✅ Pesos Phase 1 cargados")
        time_p1 = 0.0
        history_p1 = {"train_answer_mae": [0.0], "val_answer_mae": [0.0]}
    else:
        print(f"\n{'='*60}")
        print("FASE 1: Pre-training del Encoder")
        print("  Decoder + final_layer + p_gen CONGELADOS")
        print("  Solo encoder + answer_head reciben gradientes")
        print(f"{'='*60}")

        # Congelar todo excepto encoder y answer head
        v4_model.decoder.trainable = False
        v4_model.final_layer.trainable = False
        v4_model.p_gen_linear.trainable = False

        trainable_p1 = sum(v.numpy().size for v in v4_model.trainable_variables)
        frozen_p1 = total_params - trainable_p1
        print(f"  Trainable: {trainable_p1:,} params")
        print(f"  Frozen:    {frozen_p1:,} params")

        os.makedirs(phase1_dir, exist_ok=True)

        trainer_p1 = TransformerTrainerV4(
            model=v4_model,
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
            early_stopping_patience=0,
        )
        time_p1 = time.time() - start_p1

        print(f"\n  ── Fase 1 completada en {time_p1:.1f}s ({time_p1/60:.1f} min) ──")
        print(f"  Answer MAE final: {history_p1['train_answer_mae'][-1]:.1f}")

        v4_model.save_weights(p1_weights_path)
        print(f"  Pesos Phase 1 guardados: {p1_weights_path}")

    # ══════════════════════════════════════════════════════════
    #  FASE 2: Decoder + Copy Training + Fresh Cross-Attention
    # ══════════════════════════════════════════════════════════
    p2_weights_path = os.path.join(checkpoint_dir, "phase2_model.weights.h5")

    if os.path.exists(p2_weights_path):
        print(f"\n{'='*60}")
        print("FASE 2: Cargando pesos de Phase 2...")
        print(f"  (Phase 2 ya completada previamente)")
        print(f"{'='*60}")
        v4_model.load_weights(p2_weights_path)
        print(f"  ✅ Pesos Phase 2 cargados")
        time_p2 = 0.0
        history_p2 = {"train_loss": [0], "train_accuracy": [0], "train_answer_mae": [0],
                       "train_p_gen": [0.5],
                       "val_loss": [0], "val_accuracy": [0], "val_answer_mae": [0],
                       "val_p_gen": [0.5]}
    else:
        print(f"\n{'='*60}")
        print("FASE 2: Decoder + Copy Mechanism Training")
        print("  Cross-attention Q/K/V → Glorot uniform (rompe colapso)")
        print("  Encoder CONGELADO — preserva representaciones")
        print("  Decoder + p_gen aprenden cross-attention + copia")
        print(f"{'='*60}")

        # Descongelar decoder/final_layer/p_gen, congelar encoder
        v4_model.encoder.trainable = False
        v4_model.decoder.trainable = True
        v4_model.final_layer.trainable = True
        v4_model.p_gen_linear.trainable = True

        # Reinicializar cross-attention
        reinitialize_cross_attention(v4_model)

        trainable_p2 = sum(v.numpy().size for v in v4_model.trainable_variables)
        frozen_p2 = total_params - trainable_p2
        print(f"  Trainable: {trainable_p2:,} params (decoder + p_gen + answer head)")
        print(f"  Frozen:    {frozen_p2:,} params (encoder)")

        config_p2 = TransformerConfig(
            d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            dff=DFF, dropout_rate=DROPOUT,
            max_encoder_len=MAX_PROBLEM_LEN, max_decoder_len=MAX_SOLUTION_LEN,
            vocab_size=vocab_size, warmup_steps=P2_WARMUP_STEPS,
            label_smoothing=LABEL_SMOOTHING, epochs=P2_EPOCHS,
            batch_size=BATCH_SIZE, checkpoint_dir=CHECKPOINT_DIR,
        )

        # Guardar config
        config_path = os.path.join(checkpoint_dir, "config.json")
        config_p2.save(config_path)

        # Limpiar checkpoints
        for f in os.listdir(checkpoint_dir):
            if f.startswith("ckpt-") or f == "checkpoint":
                os.remove(os.path.join(checkpoint_dir, f))

        trainer_p2 = TransformerTrainerV4(
            model=v4_model,
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
        print(f"  p_gen final: {history_p2['train_p_gen'][-1]:.3f}")
        if history_p2['val_loss']:
            print(f"  Val Loss: {history_p2['val_loss'][-1]:.4f}")
            print(f"  Val Acc: {history_p2['val_accuracy'][-1]:.4f}")
            print(f"  Val p_gen: {history_p2['val_p_gen'][-1]:.3f}")

        v4_model.save_weights(p2_weights_path)
        print(f"  Pesos Phase 2 guardados: {p2_weights_path}")

    # ══════════════════════════════════════════════════════════
    #  FASE 3: Fine-tuning todo (encoder + decoder + copy)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FASE 3: Fine-tuning con Encoder Descongelado")
    print("  Todos los parámetros entrenables")
    print("  Copy mechanism + encoder se co-adaptan")
    print(f"  LR scale: {P3_LR_SCALE} → low learning rate")
    print(f"{'='*60}")

    # Descongelar todo
    v4_model.encoder.trainable = True
    v4_model.decoder.trainable = True
    v4_model.final_layer.trainable = True
    v4_model.p_gen_linear.trainable = True

    trainable_p3 = sum(v.numpy().size for v in v4_model.trainable_variables)
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

    # Limpiar checkpoints
    for f in os.listdir(checkpoint_dir):
        if f.startswith("ckpt-") or f == "checkpoint":
            os.remove(os.path.join(checkpoint_dir, f))

    trainer_p3 = TransformerTrainerV4(
        model=v4_model,
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
        early_stopping_patience=0,
    )
    time_p3 = time.time() - start_p3

    print(f"\n  ── Fase 3 completada en {time_p3:.1f}s ({time_p3/60:.1f} min) ──")
    print(f"  Loss final: {history_p3['train_loss'][-1]:.4f}")
    print(f"  Accuracy final: {history_p3['train_accuracy'][-1]:.4f}")
    print(f"  p_gen final: {history_p3['train_p_gen'][-1]:.3f}")
    if history_p3['val_loss']:
        print(f"  Val Loss: {history_p3['val_loss'][-1]:.4f}")
        print(f"  Val Acc: {history_p3['val_accuracy'][-1]:.4f}")
        print(f"  Val p_gen: {history_p3['val_p_gen'][-1]:.3f}")

    total_time = time_p1 + time_p2 + time_p3
    print(f"\n  Tiempo total (3 fases): {total_time:.1f}s ({total_time/60:.1f} min)")

    # ══════════════════════════════════════════════════════════
    #  GUARDAR MODELO FINAL
    # ══════════════════════════════════════════════════════════
    weights_path = os.path.join(checkpoint_dir, "model_weights.weights.h5")
    v4_model.save_weights(weights_path)
    print(f"\nPesos guardados en {weights_path}")

    # Copiar tokenizer
    src_tok = os.path.join(SCRIPT_DIR, TOKENIZER_MODEL)
    dst_tok = os.path.join(checkpoint_dir, "sp_tokenizer.model")
    if os.path.exists(src_tok):
        shutil.copy2(src_tok, dst_tok)
        print(f"  Tokenizer copiado a {dst_tok}")

    # Guardar historial
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

    # Guardar config v4
    v4_config = {
        "version": "v4",
        "mechanism": "pointer-generator",
        "num_copy_layers": NUM_COPY_LAYERS,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "dff": DFF,
        "vocab_size": vocab_size,
        "max_encoder_len": MAX_PROBLEM_LEN,
        "max_decoder_len": MAX_SOLUTION_LEN,
        "dropout_rate": DROPOUT,
        "answer_scale": ANSWER_SCALE,
        "total_params": total_params,
    }
    v4_config_path = os.path.join(checkpoint_dir, "config_v4.json")
    with open(v4_config_path, "w") as f:
        json.dump(v4_config, f, indent=2)

    # ══════════════════════════════════════════════════════════
    #  PROBAR INFERENCIA
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PASO FINAL: Probando inferencia V4 (Pointer-Generator)...")
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
            answer = generate_text_v4(
                v4_model, tokenizer, problem,
                max_length=MAX_SOLUTION_LEN,
                temperature=0.3,
                top_k=10,
                repetition_penalty=1.3,
            )
        print(f"  P: {problem}")
        print(f"  R: {answer[:200]}")
        print()

    print("=" * 60)
    print("  ¡ENTRENAMIENTO V4 (Pointer-Generator) COMPLETADO!")
    print("=" * 60)
    print(f"\nArchivos en: {checkpoint_dir}/")
    print(f"  - config.json / config_v4.json")
    print(f"  - sp_tokenizer.model")
    print(f"  - model_weights.weights.h5")
    print(f"  - training_history.json")


if __name__ == "__main__":
    main()
