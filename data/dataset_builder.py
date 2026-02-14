"""
dataset_builder.py — Construcción de tf.data.Dataset.

Crea el pipeline de datos para entrenamiento y evaluación del Transformer,
convirtiendo problemas JSON en tensores con padding y batching.

Uso:
    from transformer_math_physics_tutor.data.dataset_builder import create_dataset
    from transformer_math_physics_tutor.data.tokenizer import CharTokenizer

    tokenizer = CharTokenizer("vocab.json")
    dataset = create_dataset("math_clean.json", tokenizer)
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, List, Dict

from transformer_math_physics_tutor.data.tokenizer import CharTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def pad_sequence(
    sequence: List[int],
    max_length: int,
    pad_value: int = 0
) -> List[int]:
    """
    Aplica padding o truncamiento a una secuencia.

    Args:
        sequence: Lista de índices de tokens.
        max_length: Longitud máxima deseada.
        pad_value: Valor de padding (por defecto 0 = <PAD>).

    Returns:
        Secuencia con padding/truncada a max_length.
    """
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return sequence + [pad_value] * (max_length - len(sequence))


def prepare_sequences(
    problems: List[Dict],
    tokenizer: CharTokenizer,
    max_problem_len: int = 100,
    max_solution_len: int = 150
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepara las secuencias de encoder y decoder a partir de los problemas.

    Para cada problema:
    - encoder_input: tokenización del problema (con <START> y <END>)
    - decoder_input: tokenización de la solución SIN el último token (<END>)
    - decoder_target: tokenización de la solución SIN el primer token (<START>)

    Esto implementa teacher forcing: el decoder recibe como input la solución
    desplazada un paso, y el target es la solución desplazada al otro lado.

    Args:
        problems: Lista de diccionarios con 'problem' y 'solution'.
        tokenizer: Instancia de CharTokenizer con vocabulario construido.
        max_problem_len: Longitud máxima para secuencias del encoder.
        max_solution_len: Longitud máxima para secuencias del decoder.

    Returns:
        Tupla de (encoder_inputs, decoder_inputs, decoder_targets) como numpy arrays.
    """
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []

    skipped = 0

    for prob in problems:
        problem_text = prob["problem"]
        solution_text = prob["solution"]

        # Tokenizar problema (encoder input)
        enc_tokens = tokenizer.encode(problem_text, add_special_tokens=True)

        # Tokenizar solución completa (con START y END)
        sol_tokens = tokenizer.encode(solution_text, add_special_tokens=True)

        # Verificar longitudes mínimas
        if len(sol_tokens) < 2:
            skipped += 1
            continue

        # decoder_input = solución sin último token (sin END)
        dec_inp_tokens = sol_tokens[:-1]
        # decoder_target = solución sin primer token (sin START)
        dec_tar_tokens = sol_tokens[1:]

        # Aplicar padding
        enc_padded = pad_sequence(enc_tokens, max_problem_len, tokenizer.pad_token_id)
        dec_inp_padded = pad_sequence(dec_inp_tokens, max_solution_len, tokenizer.pad_token_id)
        dec_tar_padded = pad_sequence(dec_tar_tokens, max_solution_len, tokenizer.pad_token_id)

        encoder_inputs.append(enc_padded)
        decoder_inputs.append(dec_inp_padded)
        decoder_targets.append(dec_tar_padded)

    if skipped > 0:
        print(f"  Secuencias omitidas (muy cortas): {skipped}")

    encoder_inputs = np.array(encoder_inputs, dtype=np.int32)
    decoder_inputs = np.array(decoder_inputs, dtype=np.int32)
    decoder_targets = np.array(decoder_targets, dtype=np.int32)

    print(f"  Secuencias preparadas: {len(encoder_inputs)}")
    print(f"  Shape encoder_inputs: {encoder_inputs.shape}")
    print(f"  Shape decoder_inputs: {decoder_inputs.shape}")
    print(f"  Shape decoder_targets: {decoder_targets.shape}")

    return encoder_inputs, decoder_inputs, decoder_targets


def create_dataset(
    data_file: str,
    tokenizer: CharTokenizer,
    max_problem_len: int = 100,
    max_solution_len: int = 150,
    batch_size: int = 32,
    shuffle: bool = True,
    buffer_size: int = 10000
) -> tf.data.Dataset:
    """
    Crea un tf.data.Dataset listo para entrenamiento.

    El dataset tiene la estructura:
        ((encoder_input, decoder_input), decoder_target)

    Donde:
        - encoder_input: (batch, max_problem_len) — problema tokenizado
        - decoder_input: (batch, max_solution_len) — solución desplazada (teacher forcing)
        - decoder_target: (batch, max_solution_len) — target para loss

    Args:
        data_file: Ruta al archivo JSON con los problemas.
        tokenizer: Instancia de CharTokenizer con vocabulario construido.
        max_problem_len: Longitud máxima del problema (encoder).
        max_solution_len: Longitud máxima de la solución (decoder).
        batch_size: Tamaño del batch.
        shuffle: Si True, mezcla los datos.
        buffer_size: Tamaño del buffer para shuffle.

    Returns:
        tf.data.Dataset con estructura ((enc_input, dec_input), dec_target).
    """
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = DATA_DIR / data_path

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")

    # Cargar problemas
    with open(data_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print(f"Cargados {len(problems)} problemas desde {data_path}")

    # Preparar secuencias
    encoder_inputs, decoder_inputs, decoder_targets = prepare_sequences(
        problems, tokenizer, max_problem_len, max_solution_len
    )

    # Crear tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        (encoder_inputs, decoder_inputs),  # Inputs: (encoder, decoder)
        decoder_targets                     # Target
    ))

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print(f"  Dataset creado: batch_size={batch_size}, "
          f"num_batches={len(encoder_inputs) // batch_size + 1}")

    return dataset


def create_train_val_datasets(
    data_file: str = "math_clean.json",
    tokenizer: Optional[CharTokenizer] = None,
    max_problem_len: int = 100,
    max_solution_len: int = 150,
    batch_size: int = 32,
    val_split: float = 0.1,
    build_vocab: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, CharTokenizer]:
    """
    Crea datasets de entrenamiento y validación, y opcionalmente construye el tokenizer.

    Args:
        data_file: Nombre o ruta del archivo JSON con problemas.
        tokenizer: Tokenizer existente. Si None, se crea uno nuevo.
        max_problem_len: Longitud máxima del encoder.
        max_solution_len: Longitud máxima del decoder.
        batch_size: Tamaño del batch.
        val_split: Fracción de datos para validación.
        build_vocab: Si True y tokenizer es nuevo, construye el vocabulario.

    Returns:
        Tupla de (train_dataset, val_dataset, tokenizer).
    """
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = DATA_DIR / data_path

    # Cargar datos
    with open(data_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    # Crear o usar tokenizer
    if tokenizer is None:
        tokenizer = CharTokenizer()

    if build_vocab:
        all_texts = (
            [p["problem"] for p in problems] +
            [p["solution"] for p in problems]
        )
        tokenizer.build_vocab(all_texts)

    # Barajar con semilla fija para reproducibilidad
    import random
    rng = random.Random(42)
    problems_shuffled = problems.copy()
    rng.shuffle(problems_shuffled)

    # Dividir en train/val
    n_val = max(1, int(len(problems_shuffled) * val_split))
    n_train = len(problems_shuffled) - n_val

    train_problems = problems_shuffled[:n_train]
    val_problems = problems_shuffled[n_train:]

    print(f"\nDivisión train/val: {n_train}/{n_val}")

    # Crear datasets directamente en memoria (sin archivos temporales)
    print("\nCreando dataset de entrenamiento:")
    train_enc, train_dec_inp, train_dec_tar = prepare_sequences(
        train_problems, tokenizer, max_problem_len, max_solution_len
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((
        (train_enc, train_dec_inp), train_dec_tar
    )).shuffle(10000).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    print("\nCreando dataset de validación:")
    val_enc, val_dec_inp, val_dec_tar = prepare_sequences(
        val_problems, tokenizer, max_problem_len, max_solution_len
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((
        (val_enc, val_dec_inp), val_dec_tar
    )).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, tokenizer


def create_datasets_from_combined(
    data_file: str = "combined_math_physics.json",
    tokenizer: Optional[CharTokenizer] = None,
    max_problem_len: int = 200,
    max_solution_len: int = 300,
    batch_size: int = 32,
    build_vocab: bool = True,
    val_fallback_ratio: float = 0.1,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, CharTokenizer]:
    """
    Crea datasets train/val/test desde un archivo con campo 'split'.

    El archivo JSON debe contener entradas con el esquema unificado
    (ver data/schema.py), incluyendo el campo "split" = train|val|test.

    Args:
        data_file: Ruta al JSON combinado.
        tokenizer: Tokenizer existente. Si None, se crea uno nuevo.
        max_problem_len: Longitud máxima del encoder.
        max_solution_len: Longitud máxima del decoder.
        batch_size: Tamaño del batch.
        build_vocab: Si True, construye vocabulario desde los datos.
        val_fallback_ratio: Si no hay split "val" en los datos,
                             usa esta fracción del train como val.

    Returns:
        (train_dataset, val_dataset, test_dataset, tokenizer)
        test_dataset puede ser None si no hay datos de test.
    """
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = DATA_DIR / data_path

    with open(data_path, "r", encoding="utf-8") as f:
        all_problems = json.load(f)

    print(f"\nCargados {len(all_problems)} problemas desde {data_path.name}")

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
    from collections import Counter
    train_domains = Counter(p.get("domain", "?") for p in train_probs)
    val_domains = Counter(p.get("domain", "?") for p in val_probs)
    test_domains = Counter(p.get("domain", "?") for p in test_probs)

    print(f"  Train: {len(train_probs)} ({dict(train_domains)})")
    print(f"  Val:   {len(val_probs)} ({dict(val_domains)})")
    print(f"  Test:  {len(test_probs)} ({dict(test_domains)})")

    # Crear/usar tokenizer
    if tokenizer is None:
        tokenizer = CharTokenizer()

    if build_vocab:
        all_texts = (
            [p["problem"] for p in all_problems] +
            [p["solution"] for p in all_problems]
        )
        tokenizer.build_vocab(all_texts)

    # Crear datasets
    print("\nCreando dataset de entrenamiento:")
    train_enc, train_dec_inp, train_dec_tar = prepare_sequences(
        train_probs, tokenizer, max_problem_len, max_solution_len
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((
        (train_enc, train_dec_inp), train_dec_tar
    )).shuffle(10000).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    print("\nCreando dataset de validación:")
    val_enc, val_dec_inp, val_dec_tar = prepare_sequences(
        val_probs, tokenizer, max_problem_len, max_solution_len
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((
        (val_enc, val_dec_inp), val_dec_tar
    )).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    # Test (puede ser vacío)
    test_dataset = None
    if test_probs:
        print("\nCreando dataset de test:")
        test_enc, test_dec_inp, test_dec_tar = prepare_sequences(
            test_probs, tokenizer, max_problem_len, max_solution_len
        )
        test_dataset = tf.data.Dataset.from_tensor_slices((
            (test_enc, test_dec_inp), test_dec_tar
        )).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, tokenizer


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Dataset Builder")
    print("=" * 60)

    # Intentar primero el dataset combinado
    combined_path = DATA_DIR / "combined_math_physics.json"
    clean_path = DATA_DIR / "math_clean.json"

    if combined_path.exists():
        print(f"\nUsando dataset combinado: {combined_path.name}")
        train_ds, val_ds, test_ds, tok = create_datasets_from_combined()

        for (enc, dec), target in train_ds.take(1):
            print(f"\nBatch de ejemplo:")
            print(f"  Encoder input shape: {enc.shape}")
            print(f"  Decoder input shape: {dec.shape}")
            print(f"  Target shape: {target.shape}")
            print(f"\n  Primer problema: '{tok.decode(enc[0].numpy())[:80]}...'")
            print(f"  Primer target: '{tok.decode(target[0].numpy())[:80]}...'")

    elif clean_path.exists():
        print(f"\nUsando dataset legacy: {clean_path.name}")
        train_ds, val_ds, tok = create_train_val_datasets()

        for (enc, dec), target in train_ds.take(1):
            print(f"\nBatch de ejemplo:")
            print(f"  Encoder input shape: {enc.shape}")
            print(f"  Decoder input shape: {dec.shape}")
            print(f"  Target shape: {target.shape}")
    else:
        print("No se encontró ningún dataset.")
        print("Ejecuta: python data/build_combined_dataset.py")
