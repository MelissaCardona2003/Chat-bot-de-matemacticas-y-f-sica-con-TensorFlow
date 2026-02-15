"""
dataset_builder.py — Pipeline de datos con answer_value.

Estructura del dataset:
    ((encoder_input, decoder_input), (decoder_target, answer_value))

Donde answer_value es un float32 escalar.

Uso:
    from transformer_math_physics_tutor.data.dataset_builder import (
        create_datasets
    )

    train_ds, val_ds, test_ds, tokenizer = create_datasets()
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from collections import Counter

from transformer_math_physics_tutor.data.subword_tokenizer import SubwordTokenizer


def pad_sequence(
    sequence: List[int],
    max_length: int,
    pad_value: int = 0
) -> List[int]:
    """Aplica padding o truncamiento a una secuencia."""
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + [pad_value] * (max_length - len(sequence))


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def prepare_sequences_v3(
    problems: List[Dict],
    tokenizer: SubwordTokenizer,
    max_problem_len: int = 128,
    max_solution_len: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepara secuencias + answer_value para v3_easy.

    Similar a prepare_sequences_subword() pero también extrae
    answer_value de cada problema.

    Args:
        problems: Lista de dicts con 'problem', 'solution', 'answer_value'.
        tokenizer: SubwordTokenizer cargado.
        max_problem_len: Longitud máxima encoder (tokens).
        max_solution_len: Longitud máxima decoder (tokens).

    Returns:
        (encoder_inputs, decoder_inputs, decoder_targets, answer_values)
        Todos np.ndarray. answer_values shape (n,) dtype float32.
    """
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    answer_values = []

    skipped = 0
    truncated_enc = 0
    truncated_dec = 0

    for prob in problems:
        problem_text = prob["problem"]
        solution_text = prob["solution"]
        answer_val = prob.get("answer_value", 0.0)

        # Tokenizar
        enc_tokens = tokenizer.encode(problem_text, add_special_tokens=True)
        sol_tokens = tokenizer.encode(solution_text, add_special_tokens=True)

        if len(sol_tokens) < 2:
            skipped += 1
            continue

        if len(enc_tokens) > max_problem_len:
            truncated_enc += 1
        if len(sol_tokens) > max_solution_len + 1:
            truncated_dec += 1

        dec_inp_tokens = sol_tokens[:-1]
        dec_tar_tokens = sol_tokens[1:]

        enc_padded = pad_sequence(enc_tokens, max_problem_len, tokenizer.pad_token_id)
        dec_inp_padded = pad_sequence(dec_inp_tokens, max_solution_len, tokenizer.pad_token_id)
        dec_tar_padded = pad_sequence(dec_tar_tokens, max_solution_len, tokenizer.pad_token_id)

        encoder_inputs.append(enc_padded)
        decoder_inputs.append(dec_inp_padded)
        decoder_targets.append(dec_tar_padded)
        answer_values.append(float(answer_val))

    if skipped > 0:
        print(f"  Secuencias omitidas (muy cortas): {skipped}")
    if truncated_enc > 0:
        print(f"  Encoder truncados: {truncated_enc}/{len(problems)} "
              f"({truncated_enc/len(problems)*100:.1f}%)")
    if truncated_dec > 0:
        print(f"  Decoder truncados: {truncated_dec}/{len(problems)} "
              f"({truncated_dec/len(problems)*100:.1f}%)")

    encoder_inputs = np.array(encoder_inputs, dtype=np.int32)
    decoder_inputs = np.array(decoder_inputs, dtype=np.int32)
    decoder_targets = np.array(decoder_targets, dtype=np.int32)
    answer_values = np.array(answer_values, dtype=np.float32)

    print(f"  Secuencias preparadas: {len(encoder_inputs)}")
    print(f"  Shape encoder_inputs: {encoder_inputs.shape}")
    print(f"  Shape decoder_inputs: {decoder_inputs.shape}")
    print(f"  Shape decoder_targets: {decoder_targets.shape}")
    print(f"  Shape answer_values: {answer_values.shape}")
    print(f"  Answer range: [{answer_values.min():.1f}, {answer_values.max():.1f}]")

    return encoder_inputs, decoder_inputs, decoder_targets, answer_values


def create_datasets_v3_easy(
    data_file: str = "combined_easy.json",
    tokenizer_model: str = "checkpoints/v2_subword/sp_tokenizer.model",
    max_problem_len: int = 128,
    max_solution_len: int = 256,
    batch_size: int = 32,
    val_fallback_ratio: float = 0.1,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset], SubwordTokenizer]:
    """
    Crea datasets train/val/test para v3_easy con answer_value.

    Estructura del dataset:
        ((encoder_input, decoder_input), (decoder_target, answer_value))

    Args:
        data_file: Ruta al JSON con problemas fáciles (con answer_value).
        tokenizer_model: Ruta al .model de SentencePiece.
        max_problem_len: Longitud máxima encoder (tokens subword).
        max_solution_len: Longitud máxima decoder (tokens subword).
        batch_size: Tamaño del batch.
        val_fallback_ratio: Fracción para val si no hay split "val".

    Returns:
        (train_dataset, val_dataset, test_dataset, tokenizer)
    """
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = DATA_DIR / data_path

    with open(data_path, "r", encoding="utf-8") as f:
        all_problems = json.load(f)

    print(f"\nCargados {len(all_problems)} problemas fáciles desde {data_path.name}")

    # Separar por split
    train_probs = [p for p in all_problems if p.get("split") == "train"]
    val_probs = [p for p in all_problems if p.get("split") == "val"]
    test_probs = [p for p in all_problems if p.get("split") == "test"]

    # Fallback: si no hay val, separar del train
    if not val_probs and train_probs:
        import random
        rng = random.Random(42)
        rng.shuffle(train_probs)
        n_val = max(1, int(len(train_probs) * val_fallback_ratio))
        val_probs = train_probs[:n_val]
        train_probs = train_probs[n_val:]

    # Contar por dominio
    train_domains = Counter(p.get("domain", "?") for p in train_probs)
    val_domains = Counter(p.get("domain", "?") for p in val_probs)
    test_domains = Counter(p.get("domain", "?") for p in test_probs)

    print(f"  Train: {len(train_probs)} ({dict(train_domains)})")
    print(f"  Val:   {len(val_probs)} ({dict(val_domains)})")
    print(f"  Test:  {len(test_probs)} ({dict(test_domains)})")

    # Cargar tokenizer
    tokenizer_path = Path(tokenizer_model)
    if not tokenizer_path.is_absolute():
        tokenizer_path = BASE_DIR / tokenizer_model
    tokenizer = SubwordTokenizer(str(tokenizer_path))

    # Crear datos
    print(f"\nCreando dataset de entrenamiento (v3_easy):")
    train_enc, train_dec_inp, train_dec_tar, train_ans = prepare_sequences_v3(
        train_probs, tokenizer, max_problem_len, max_solution_len
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((
        (train_enc, train_dec_inp), (train_dec_tar, train_ans)
    )).shuffle(10000).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    print(f"\nCreando dataset de validación (v3_easy):")
    val_enc, val_dec_inp, val_dec_tar, val_ans = prepare_sequences_v3(
        val_probs, tokenizer, max_problem_len, max_solution_len
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((
        (val_enc, val_dec_inp), (val_dec_tar, val_ans)
    )).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    # Test
    test_dataset = None
    if test_probs:
        print(f"\nCreando dataset de test (v3_easy):")
        test_enc, test_dec_inp, test_dec_tar, test_ans = prepare_sequences_v3(
            test_probs, tokenizer, max_problem_len, max_solution_len
        )
        test_dataset = tf.data.Dataset.from_tensor_slices((
            (test_enc, test_dec_inp), (test_dec_tar, test_ans)
        )).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, tokenizer


if __name__ == "__main__":
    import os
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.8")
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

    print("=" * 60)
    print("DEMO: Dataset Builder v3_easy")
    print("=" * 60)

    train_ds, val_ds, test_ds, tok = create_datasets_v3_easy()

    for (enc, dec), (target, answer_val) in train_ds.take(1):
        print(f"\nBatch de ejemplo:")
        print(f"  Encoder input shape: {enc.shape}")
        print(f"  Decoder input shape: {dec.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Answer values shape: {answer_val.shape}")
        print(f"  Answer values (primeros 5): {answer_val[:5].numpy()}")
        print(f"\n  Primer problema: '{tok.decode(enc[0].numpy().tolist())[:80]}...'")
