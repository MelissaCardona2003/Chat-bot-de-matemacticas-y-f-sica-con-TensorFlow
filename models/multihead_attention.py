"""
multihead_attention.py — Multi-Head Attention desde cero.

Implementa Scaled Dot-Product Attention y Multi-Head Attention
como se describe en "Attention Is All You Need" (Vaswani et al., 2017).

Uso:
    from transformer_math_physics_tutor.models.multihead_attention import MultiHeadAttention
    mha = MultiHeadAttention(d_model=256, num_heads=8)
    output, weights = mha(v, k, q, mask=None)
"""

import tensorflow as tf


def scaled_dot_product_attention(
    q: tf.Tensor,
    k: tf.Tensor,
    v: tf.Tensor,
    mask: tf.Tensor = None
) -> tuple:
    """
    Calcula la atención de producto punto escalado.

    Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

    Esta es la operación central del Transformer: cada posición
    en la query "atiende" a todas las posiciones en la key/value.

    Args:
        q: Tensor de queries, shape (..., seq_len_q, depth).
        k: Tensor de keys, shape (..., seq_len_k, depth).
        v: Tensor de values, shape (..., seq_len_v, depth).
           Nota: seq_len_k == seq_len_v siempre.
        mask: Tensor de máscara opcional, shape compatible para broadcasting.
              Valores 1.0 en posiciones a enmascarar.

    Returns:
        Tupla de (output, attention_weights):
        - output: shape (..., seq_len_q, depth)
        - attention_weights: shape (..., seq_len_q, seq_len_k)
    """
    # Producto punto de Q y K^T: (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # Escalar por sqrt(d_k) para estabilidad numérica
    # Sin esto, el softmax tendría gradientes muy pequeños para d_k grande
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Aplicar máscara: poner -1e9 en posiciones enmascaradas
    # Esto hace que softmax asigne ~0 probabilidad a esas posiciones
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax en la última dimensión (seq_len_k)
    # Cada query obtiene una distribución de probabilidad sobre las keys
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiplicar por values: weighted sum
    # (..., seq_len_q, seq_len_k) x (..., seq_len_v, depth) → (..., seq_len_q, depth)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-Head Attention Layer.

    En lugar de una sola atención con d_model dimensiones,
    divide en num_heads cabezas con depth = d_model/num_heads cada una.
    Esto permite al modelo atender a información de diferentes
    subespacios de representación en distintas posiciones.

    MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W^O
    donde head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

    Attributes:
        num_heads: Número de cabezas de atención.
        d_model: Dimensión del modelo.
        depth: Dimensión por cabeza (d_model // num_heads).
        wq: Capa Dense para proyectar queries.
        wk: Capa Dense para proyectar keys.
        wv: Capa Dense para proyectar values.
        dense: Capa Dense final (proyección de salida).
    """

    def __init__(self, d_model: int, num_heads: int, **kwargs):
        """
        Inicializa la capa de Multi-Head Attention.

        Args:
            d_model: Dimensión del modelo.
            num_heads: Número de cabezas de atención.

        Raises:
            AssertionError: Si d_model no es divisible por num_heads.
        """
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.supports_masking = True
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) debe ser divisible por num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        # Capas de proyección lineal para Q, K, V
        # Cada una proyecta de d_model → d_model
        self.wq = tf.keras.layers.Dense(d_model)  # W^Q
        self.wk = tf.keras.layers.Dense(d_model)  # W^K
        self.wv = tf.keras.layers.Dense(d_model)  # W^V

        # Proyección de salida: concatenación de cabezas → d_model
        self.dense = tf.keras.layers.Dense(d_model)  # W^O

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """
        Divide la última dimensión en (num_heads, depth) y transpone.

        Convierte de (batch, seq_len, d_model) a (batch, num_heads, seq_len, depth)
        para procesar cada cabeza en paralelo.

        Args:
            x: Tensor de shape (batch_size, seq_len, d_model).
            batch_size: Tamaño del batch actual.

        Returns:
            Tensor de shape (batch_size, num_heads, seq_len, depth).
        """
        # (batch, seq_len, d_model) → (batch, seq_len, num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # (batch, seq_len, num_heads, depth) → (batch, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self,
        v: tf.Tensor,
        k: tf.Tensor,
        q: tf.Tensor,
        mask: tf.Tensor = None
    ) -> tuple:
        """
        Forward pass de Multi-Head Attention.

        Args:
            v: Values, shape (batch_size, seq_len_v, d_model).
            k: Keys, shape (batch_size, seq_len_k, d_model).
            q: Queries, shape (batch_size, seq_len_q, d_model).
            mask: Máscara opcional, shape compatible para broadcasting.

        Returns:
            Tupla de (output, attention_weights):
            - output: shape (batch_size, seq_len_q, d_model)
            - attention_weights: shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = tf.shape(q)[0]

        # Paso 1: Proyectar Q, K, V con capas Dense
        # (batch, seq_len, d_model) → (batch, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Paso 2: Dividir en múltiples cabezas
        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Paso 3: Scaled dot-product attention (por cada cabeza en paralelo)
        # output: (batch, num_heads, seq_len_q, depth)
        # weights: (batch, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        # Paso 4: Transponer y concatenar cabezas
        # (batch, num_heads, seq_len_q, depth) → (batch, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch, seq_len_q, num_heads, depth) → (batch, seq_len_q, d_model)
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )

        # Paso 5: Proyección final
        # (batch, seq_len_q, d_model) → (batch, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights

    def get_config(self):
        """Serialización de la configuración de la capa."""
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
        })
        return config


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Multi-Head Attention")
    print("=" * 60)

    # Crear capa MHA
    mha = MultiHeadAttention(d_model=256, num_heads=8)

    # Crear tensores de prueba (batch=2, seq_len=10, d_model=256)
    batch_size = 2
    seq_len = 10
    d_model = 256

    q = tf.random.normal((batch_size, seq_len, d_model))
    k = tf.random.normal((batch_size, seq_len, d_model))
    v = tf.random.normal((batch_size, seq_len, d_model))

    output, attn_weights = mha(v, k, q, mask=None)

    print(f"Input shape: ({batch_size}, {seq_len}, {d_model})")
    print(f"Output shape: {output.shape}")           # (2, 10, 256)
    print(f"Attention weights shape: {attn_weights.shape}")  # (2, 8, 10, 10)
    print(f"Attention weights sum (should be ~1): "
          f"{tf.reduce_sum(attn_weights[0, 0, 0, :]).numpy():.4f}")
