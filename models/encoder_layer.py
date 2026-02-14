"""
encoder_layer.py — Capa de Encoder del Transformer.

Implementa una capa del encoder con:
1. Multi-Head Self-Attention
2. Feed-Forward Network
3. Residual connections + Layer Normalization

Uso:
    from transformer_math_physics_tutor.models.encoder_layer import EncoderLayer
    layer = EncoderLayer(d_model=256, num_heads=8, dff=1024, rate=0.1)
"""

import tensorflow as tf
from transformer_math_physics_tutor.models.multihead_attention import MultiHeadAttention
from transformer_math_physics_tutor.models.xla_dropout import XLADropout


def point_wise_feed_forward_network(d_model: int, dff: int) -> tf.keras.Sequential:
    """
    Crea una red feed-forward de dos capas Dense.

    FFN(x) = max(0, x·W1 + b1)·W2 + b2

    La primera capa expande la dimensión de d_model a dff (típicamente 4x),
    aplica ReLU, y la segunda vuelve a contraer a d_model.

    Args:
        d_model: Dimensión del modelo (entrada y salida).
        dff: Dimensión interna de la capa feed-forward.

    Returns:
        tf.keras.Sequential con dos capas Dense.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),   # (batch, seq, dff)
        tf.keras.layers.Dense(d_model)                    # (batch, seq, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    """
    Una capa del Encoder del Transformer.

    Flujo:
        input → Self-Attention → Add & Norm → FFN → Add & Norm → output

    Cada sub-capa tiene una conexión residual y normalización de capa:
        LayerNorm(x + Sublayer(x))

    Attributes:
        mha: Capa de Multi-Head Self-Attention.
        ffn: Red Feed-Forward de dos capas.
        layernorm1: Layer Normalization después de attention.
        layernorm2: Layer Normalization después de FFN.
        dropout1: Dropout después de attention.
        dropout2: Dropout después de FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        rate: float = 0.1,
        **kwargs
    ):
        """
        Inicializa la capa de encoder.

        Args:
            d_model: Dimensión del modelo.
            num_heads: Número de cabezas de atención.
            dff: Dimensión interna de la FFN.
            rate: Tasa de dropout.
        """
        super(EncoderLayer, self).__init__(**kwargs)

        self.supports_masking = True

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = XLADropout(rate)
        self.dropout2 = XLADropout(rate)

    def call(
        self,
        x: tf.Tensor,
        training: bool = False,
        mask: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Forward pass de la capa de encoder.

        Args:
            x: Tensor de entrada, shape (batch_size, seq_len, d_model).
            training: Si True, aplica dropout.
            mask: Máscara de padding, shape (batch, 1, 1, seq_len).

        Returns:
            Tensor de salida, shape (batch_size, seq_len, d_model).
        """
        # Sub-capa 1: Multi-Head Self-Attention
        # En self-attention, Q = K = V = x
        attn_output, _ = self.mha(x, x, x, mask)  # (batch, seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        # Conexión residual + Layer Normalization
        out1 = self.layernorm1(x + attn_output)    # (batch, seq_len, d_model)

        # Sub-capa 2: Feed-Forward Network
        ffn_output = self.ffn(out1)                 # (batch, seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Conexión residual + Layer Normalization
        out2 = self.layernorm2(out1 + ffn_output)  # (batch, seq_len, d_model)

        return out2

    def get_config(self):
        """Serialización de la configuración."""
        config = super().get_config()
        config.update({
            "d_model": self.mha.d_model,
            "num_heads": self.mha.num_heads,
            "dff": self.ffn.layers[0].units,
            "rate": self.dropout1.rate,
        })
        return config


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Encoder Layer")
    print("=" * 60)

    layer = EncoderLayer(d_model=256, num_heads=8, dff=1024, rate=0.1)

    # Input de prueba: (batch=2, seq_len=10, d_model=256)
    x = tf.random.normal((2, 10, 256))
    output = layer(x, training=False)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # (2, 10, 256)
    print("✓ Las dimensiones se mantienen (residual connection)")
