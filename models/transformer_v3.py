"""
transformer_v3.py — Transformer con cabeza de regresión de respuesta.

Extiende el Transformer base (v1/v2) con un MLP que predice el valor
numérico de la respuesta a partir del decoder output (mean-pooled).

Esta cabeza auxiliar crea un gradiente DIRECTO desde la respuesta
numérica → decoder → cross-attention → encoder, forzando al modelo
a realmente LEER el problema.

Uso:
    from transformer_math_physics_tutor.models.transformer_v3 import TransformerV3

    config = TransformerConfig(vocab_size=4000)
    model = TransformerV3(config, answer_scale=1000.0)
    logits, answer_pred = model((enc_input, dec_input), training=True)
"""

import tensorflow as tf
from transformer_math_physics_tutor.models.transformer import Transformer
from transformer_math_physics_tutor.models.config import TransformerConfig
from transformer_math_physics_tutor.models.xla_dropout import XLADropout


class TransformerV3(Transformer):
    """
    Transformer v3 con cabeza auxiliar de regresión numérica.

    Hereda toda la arquitectura del Transformer base (encoder, decoder,
    final_layer) y añade un answer_head MLP que predice el valor
    numérico de la respuesta.

    Durante training:
        loss_total = loss_seq2seq + λ * loss_answer

    Durante inference:
        Solo se usa la salida seq2seq (logits). answer_pred se ignora.

    Attributes:
        answer_scale: Factor de escala para normalizar answer_value.
                      Si answer_value original está en [-10000, 10000],
                      dividimos por 1000 → head predice en [-10, 10].
        answer_head: MLP que proyecta decoder_output → scalar.
    """

    def __init__(self, config: TransformerConfig, answer_scale: float = 1000.0, **kwargs):
        """
        Inicializa TransformerV3.

        Args:
            config: Configuración del Transformer.
            answer_scale: Divisor para normalizar answer_value.
        """
        super().__init__(config, **kwargs)

        self.answer_scale = answer_scale

        # MLP: decoder_pooled (d_model) → scalar
        # Dos capas Dense con ReLU + Dropout para regularización
        self.answer_dense1 = tf.keras.layers.Dense(
            config.d_model, activation='relu', name='answer_dense1'
        )
        self.answer_dropout = XLADropout(0.1, name='answer_dropout')
        self.answer_dense2 = tf.keras.layers.Dense(
            1, name='answer_dense2'
        )

    def call(
        self,
        inputs: tuple,
        training: bool = False,
        return_attention: bool = False
    ):
        """
        Forward pass con predicción de respuesta numérica.

        Args:
            inputs: Tupla de (inp, tar).
            training: Si True, aplica dropout.
            return_attention: Si True, retorna attention_weights.

        Returns:
            Si return_attention=False:
                Tupla de (logits, answer_pred).
                - logits: shape (batch, tar_seq_len, vocab_size)
                - answer_pred: shape (batch,) — valor numérico predicho (escalado)
            Si return_attention=True:
                Tupla de (logits, answer_pred, attention_weights).
        """
        inp, tar = inputs

        # --- Crear máscaras (mismo que Transformer base) ---
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        # --- Encoder ---
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)

        # --- Decoder ---
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training=training,
            look_ahead_mask=combined_mask, padding_mask=dec_padding_mask
        )

        # --- Capa lineal final (seq2seq logits) ---
        final_output = self.final_layer(dec_output)

        # --- Answer head: usa ENCODER output (gradiente directo al encoder) ---
        # Mean-pool encoder output ignorando padding
        enc_mask = tf.cast(tf.not_equal(inp, 0), tf.float32)  # (batch, inp_seq_len)
        enc_mask_expanded = enc_mask[:, :, tf.newaxis]  # (batch, inp_seq_len, 1)
        pooled = tf.reduce_sum(enc_output * enc_mask_expanded, axis=1)  # (batch, d_model)
        n_real_enc = tf.maximum(tf.reduce_sum(enc_mask, axis=1, keepdims=True), 1.0)
        pooled = pooled / n_real_enc  # (batch, d_model)

        # MLP: d_model → d_model → 1
        answer_hidden = self.answer_dense1(pooled)  # (batch, d_model)
        answer_hidden = self.answer_dropout(answer_hidden, training=training)
        answer_pred = self.answer_dense2(answer_hidden)  # (batch, 1)
        answer_pred = tf.squeeze(answer_pred, axis=-1)  # (batch,)

        if return_attention:
            return final_output, answer_pred, attention_weights
        return final_output, answer_pred


def answer_regression_loss(
    answer_pred: tf.Tensor,
    answer_true: tf.Tensor,
    scale: float = 1000.0,
    delta: float = 1.0
) -> tf.Tensor:
    """
    Calcula Huber loss entre respuesta predicha y verdadera.

    Usa Huber loss (suave para outliers) en vez de MSE puro.

    Args:
        answer_pred: Predicción del head, shape (batch,).
        answer_true: Valor verdadero (NO escalado), shape (batch,).
        scale: Factor de escala (divide answer_true).
        delta: Umbral de Huber loss.

    Returns:
        Escalar con el loss promedio.
    """
    # Normalizar la verdad
    answer_true_scaled = answer_true / scale

    # Huber loss: cuadrática cerca de 0, lineal lejos
    error = answer_pred - answer_true_scaled
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear

    return tf.reduce_mean(loss)


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: TransformerV3 con Answer Head")
    print("=" * 60)

    config = TransformerConfig(
        d_model=64, num_heads=4, num_layers=2,
        dff=128, vocab_size=50,
        max_encoder_len=20, max_decoder_len=15,
    )

    model = TransformerV3(config, answer_scale=1000.0)

    # Forward pass
    enc = tf.random.uniform((2, 20), 1, 50, dtype=tf.int32)
    dec = tf.random.uniform((2, 15), 1, 50, dtype=tf.int32)

    logits, answer_pred = model((enc, dec), training=False)
    print(f"Logits shape: {logits.shape}")      # (2, 15, 50)
    print(f"Answer pred shape: {answer_pred.shape}")  # (2,)
    print(f"Answer pred values: {answer_pred.numpy()}")

    # Con attention
    logits, answer_pred, attn = model((enc, dec), training=False, return_attention=True)
    print(f"Attention keys: {list(attn.keys())}")

    # Test answer_regression_loss
    answer_true = tf.constant([42.0, 100.0])
    loss = answer_regression_loss(answer_pred, answer_true, scale=1000.0)
    print(f"Answer loss: {loss.numpy():.4f}")

    model.summary()
    print(f"Total params: {model.count_params():,}")
