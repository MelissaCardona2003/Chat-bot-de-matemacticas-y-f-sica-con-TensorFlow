"""
losses.py — Funciones de pérdida con máscara.

Implementa la pérdida y accuracy que ignoran tokens de padding,
esenciales para entrenamiento con secuencias de longitud variable.

Uso:
    from transformer_math_physics_tutor.training.losses import loss_function, accuracy_function
    loss = loss_function(real, pred, label_smoothing=0.1)
    acc = accuracy_function(real, pred)
"""

import tensorflow as tf


def loss_function(
    real: tf.Tensor,
    pred: tf.Tensor,
    label_smoothing: float = 0.1
) -> tf.Tensor:
    """
    Calcula la pérdida de cross-entropy con máscara de padding y label smoothing.

    Los tokens de padding (<PAD> = 0) no deben contribuir al loss,
    ya que no son parte real de la secuencia. La máscara los excluye.

    Label smoothing suaviza los targets one-hot para regularizar:
    - En lugar de [0, 0, 1, 0] usa [ε/V, ε/V, 1-ε+ε/V, ε/V]
    - Donde ε = label_smoothing, V = vocab_size
    - Recomendado por el paper "Attention Is All You Need" (ε=0.1)

    Args:
        real: Tensor de labels reales, shape (batch, seq_len).
              Contiene índices de tokens (int32).
        pred: Tensor de predicciones (logits), shape (batch, seq_len, vocab_size).
              Son logits crudos (antes de softmax).
        label_smoothing: Factor de suavizado de etiquetas (0-1).
                         0.0 = one-hot puro, 0.1 = recomendado por el paper.

    Returns:
        Escalar con la pérdida promedio (solo sobre tokens no-padding).
    """
    vocab_size = tf.shape(pred)[-1]

    # Convertir sparse labels a one-hot para aplicar label smoothing
    # real: (batch, seq_len) -> one_hot: (batch, seq_len, vocab_size)
    one_hot_labels = tf.one_hot(tf.cast(real, tf.int32), vocab_size)

    # Aplicar label smoothing: suavizar la distribución one-hot
    # smooth = (1 - ε) * one_hot + ε / V
    if label_smoothing > 0.0:
        one_hot_labels = (
            one_hot_labels * (1.0 - label_smoothing)
            + label_smoothing / tf.cast(vocab_size, tf.float32)
        )

    # Calcular cross-entropy manualmente con logits
    # loss = -sum(labels * log_softmax(logits))
    log_probs = tf.nn.log_softmax(pred, axis=-1)
    loss_ = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    # loss_ shape: (batch, seq_len)

    # Crear máscara: True donde NO hay padding, False donde hay padding
    # real != 0 → tokens reales (no padding)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Aplicar máscara: multiplicar loss por máscara (padding → 0)
    loss_ *= mask

    # Promediar solo sobre tokens válidos (no padding)
    # Esto evita que el padding diluya la señal de loss
    total_loss = tf.reduce_sum(loss_)
    num_valid = tf.reduce_sum(mask)

    # Evitar división por cero
    return total_loss / tf.maximum(num_valid, 1.0)


def accuracy_function(
    real: tf.Tensor,
    pred: tf.Tensor
) -> tf.Tensor:
    """
    Calcula la accuracy por token ignorando padding.

    Compara la predicción (argmax de logits) con el token real,
    excluyendo posiciones de padding.

    Args:
        real: Tensor de labels reales, shape (batch, seq_len).
        pred: Tensor de predicciones (logits), shape (batch, seq_len, vocab_size).

    Returns:
        Escalar con la accuracy promedio (proporción de tokens correctos).
    """
    # Obtener token predicho (argmax sobre vocabulario)
    # (batch, seq_len, vocab_size) → (batch, seq_len)
    predicted_ids = tf.argmax(pred, axis=-1)

    # Comparar con tokens reales
    # Necesitamos cast porque argmax devuelve int64 y real puede ser int32
    real_casted = tf.cast(real, dtype=predicted_ids.dtype)
    correct = tf.cast(tf.equal(predicted_ids, real_casted), dtype=tf.float32)

    # Máscara de padding: True donde hay tokens reales
    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), dtype=tf.float32)

    # Aplicar máscara y promediar
    correct *= mask

    total_correct = tf.reduce_sum(correct)
    num_valid = tf.reduce_sum(mask)

    return total_correct / tf.maximum(num_valid, 1.0)


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Loss y Accuracy Functions")
    print("=" * 60)

    # Crear datos de prueba
    vocab_size = 10
    batch_size = 2
    seq_len = 5

    # Labels reales (con padding al final)
    real = tf.constant([
        [1, 3, 5, 2, 0],  # 4 tokens + 1 padding
        [2, 4, 0, 0, 0],  # 2 tokens + 3 padding
    ], dtype=tf.int32)

    # Predicciones (logits aleatorios)
    pred = tf.random.normal((batch_size, seq_len, vocab_size))

    loss = loss_function(real, pred)
    acc = accuracy_function(real, pred)

    print(f"Real shape: {real.shape}")
    print(f"Pred shape: {pred.shape}")
    print(f"Loss (masked): {loss.numpy():.4f}")
    print(f"Accuracy (masked): {acc.numpy():.4f}")

    # Verificar con predicciones perfectas
    perfect_pred = tf.one_hot(real, vocab_size) * 100  # Logits muy altos para tokens correctos
    perfect_loss = loss_function(real, perfect_pred)
    perfect_acc = accuracy_function(real, perfect_pred)
    print(f"\nCon predicciones perfectas:")
    print(f"  Loss: {perfect_loss.numpy():.6f}")
    print(f"  Accuracy: {perfect_acc.numpy():.4f}")
