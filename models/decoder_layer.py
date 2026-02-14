"""
decoder_layer.py — Capa de Decoder del Transformer.

Implementa una capa del decoder con:
1. Masked Multi-Head Self-Attention (causal)
2. Multi-Head Cross-Attention (encoder-decoder)
3. Feed-Forward Network
4. Residual connections + Layer Normalization

Uso:
    from transformer_math_physics_tutor.models.decoder_layer import DecoderLayer
    layer = DecoderLayer(d_model=256, num_heads=8, dff=1024, rate=0.1)
"""

import tensorflow as tf
from transformer_math_physics_tutor.models.multihead_attention import MultiHeadAttention
from transformer_math_physics_tutor.models.xla_dropout import XLADropout
from transformer_math_physics_tutor.models.encoder_layer import point_wise_feed_forward_network


class DecoderLayer(tf.keras.layers.Layer):
    """
    Una capa del Decoder del Transformer.

    Flujo:
        input → Masked Self-Attention → Add & Norm
              → Cross-Attention (con encoder output) → Add & Norm
              → FFN → Add & Norm → output

    A diferencia del encoder, el decoder tiene tres sub-capas:
    1. Self-attention enmascarado (look-ahead mask) para no ver tokens futuros
    2. Cross-attention con la salida del encoder
    3. Feed-forward network

    Attributes:
        mha1: Self-attention enmascarado.
        mha2: Cross-attention (encoder-decoder).
        ffn: Red feed-forward.
        layernorm1-3: Layer Normalization para cada sub-capa.
        dropout1-3: Dropout para cada sub-capa.
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
        Inicializa la capa de decoder.

        Args:
            d_model: Dimensión del modelo.
            num_heads: Número de cabezas de atención.
            dff: Dimensión interna de la FFN.
            rate: Tasa de dropout.
        """
        super(DecoderLayer, self).__init__(**kwargs)

        # Sub-capa 1: Masked Self-Attention
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        # Sub-capa 2: Cross-Attention (encoder-decoder)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        # Sub-capa 3: Feed-Forward Network
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Layer Normalizations (una por sub-capa)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropouts (uno por sub-capa)
        self.dropout1 = XLADropout(rate)
        self.dropout2 = XLADropout(rate)
        self.dropout3 = XLADropout(rate)

    def call(
        self,
        x: tf.Tensor,
        enc_output: tf.Tensor,
        training: bool = False,
        look_ahead_mask: tf.Tensor = None,
        padding_mask: tf.Tensor = None
    ) -> tuple:
        """
        Forward pass de la capa de decoder.

        Args:
            x: Entrada al decoder, shape (batch, target_seq_len, d_model).
            enc_output: Salida del encoder, shape (batch, input_seq_len, d_model).
            training: Si True, aplica dropout.
            look_ahead_mask: Máscara triangular + padding para self-attention.
                             Previene que el decoder vea tokens futuros.
            padding_mask: Máscara de padding para cross-attention.
                          Enmascara tokens <PAD> del encoder.

        Returns:
            Tupla de (output, attn_weights_1, attn_weights_2):
            - output: shape (batch, target_seq_len, d_model)
            - attn_weights_1: Pesos de self-attention
            - attn_weights_2: Pesos de cross-attention
        """
        # Sub-capa 1: Masked Self-Attention
        # Q = K = V = x (self-attention)
        # look_ahead_mask previene ver tokens futuros
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)  # Residual + LayerNorm

        # Sub-capa 2: Cross-Attention (Encoder-Decoder)
        # Q = salida de sub-capa anterior, K = V = salida del encoder
        # Esto permite al decoder atender a todas las posiciones del encoder
        attn2, attn_weights_block2 = self.mha2(
            enc_output,  # V: valores del encoder
            enc_output,  # K: claves del encoder
            out1,        # Q: queries del decoder
            padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # Residual + LayerNorm

        # Sub-capa 3: Feed-Forward Network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # Residual + LayerNorm

        return out3, attn_weights_block1, attn_weights_block2

    def get_config(self):
        """Serialización de la configuración."""
        config = super().get_config()
        config.update({
            "d_model": self.mha1.d_model,
            "num_heads": self.mha1.num_heads,
            "dff": self.ffn.layers[0].units,
            "rate": self.dropout1.rate,
        })
        return config


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Decoder Layer")
    print("=" * 60)

    layer = DecoderLayer(d_model=256, num_heads=8, dff=1024, rate=0.1)

    # Entradas de prueba
    x = tf.random.normal((2, 15, 256))          # (batch=2, target_seq=15, d_model)
    enc_out = tf.random.normal((2, 10, 256))     # (batch=2, input_seq=10, d_model)

    output, w1, w2 = layer(x, enc_out, training=False)

    print(f"Decoder input shape: {x.shape}")
    print(f"Encoder output shape: {enc_out.shape}")
    print(f"Decoder output shape: {output.shape}")     # (2, 15, 256)
    print(f"Self-attn weights: {w1.shape}")             # (2, 8, 15, 15)
    print(f"Cross-attn weights: {w2.shape}")            # (2, 8, 15, 10)
