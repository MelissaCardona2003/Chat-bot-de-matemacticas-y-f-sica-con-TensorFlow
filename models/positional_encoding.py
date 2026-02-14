"""
positional_encoding.py — Codificación posicional sinusoidal.

Implementa el positional encoding del paper "Attention Is All You Need"
usando funciones seno y coseno para inyectar información posicional.

Fórmulas:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Uso:
    from transformer_math_physics_tutor.models.positional_encoding import positional_encoding
    pe = positional_encoding(100, 256)  # (1, 100, 256)
"""

import numpy as np
import tensorflow as tf


def get_angles(
    pos: np.ndarray,
    i: np.ndarray,
    d_model: int
) -> np.ndarray:
    """
    Calcula los ángulos para la codificación posicional.

    La fórmula es: angle = pos / 10000^(2i/d_model)
    donde pos es la posición y i es la dimensión.

    Args:
        pos: Array de posiciones, shape (position, 1).
        i: Array de dimensiones, shape (1, d_model).
        d_model: Dimensión del modelo.

    Returns:
        Array de ángulos, shape (position, d_model).
    """
    # Calcular exponentes: 2i/d_model para cada dimensión
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position: int, d_model: int) -> tf.Tensor:
    """
    Genera la codificación posicional sinusoidal.

    Para posiciones pares (2i): aplica sin()
    Para posiciones impares (2i+1): aplica cos()

    Esto permite al modelo aprender relaciones posicionales relativas,
    ya que PE(pos+k) puede expresarse como función lineal de PE(pos).

    Args:
        position: Número máximo de posiciones (longitud de secuencia).
        d_model: Dimensión del modelo (debe ser par para sin/cos).

    Returns:
        Tensor de shape (1, position, d_model) con la codificación posicional.
        El primer eje es para broadcasting con el batch.
    """
    # Crear matrices de posiciones e índices de dimensión
    # pos: (position, 1), i: (1, d_model)
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],      # (position, 1)
        np.arange(d_model)[np.newaxis, :],        # (1, d_model)
        d_model
    )

    # Aplicar sin a índices pares (0, 2, 4, ...)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Aplicar cos a índices impares (1, 3, 5, ...)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # Añadir dimensión batch: (position, d_model) → (1, position, d_model)
    pos_encoding = angle_rads[np.newaxis, :, :].astype(np.float32)

    return tf.constant(pos_encoding, dtype=tf.float32)


if __name__ == "__main__":
    # Demo: visualizar positional encoding
    print("=" * 60)
    print("DEMO: Positional Encoding")
    print("=" * 60)

    pe = positional_encoding(50, 256)
    print(f"Shape: {pe.shape}")  # (1, 50, 256)
    print(f"Rango de valores: [{tf.reduce_min(pe):.4f}, {tf.reduce_max(pe):.4f}]")
    print(f"Primeros valores posición 0: {pe[0, 0, :8].numpy()}")
    print(f"Primeros valores posición 1: {pe[0, 1, :8].numpy()}")
