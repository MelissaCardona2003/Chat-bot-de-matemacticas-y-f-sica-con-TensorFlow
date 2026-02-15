"""
transformer_v4.py — Transformer con Pointer-Generator Network (Copy Mechanism).

Extiende TransformerV3 con un mecanismo de copia inspirado en
"Get To The Point" (See et al., 2017) adaptado para Transformers.

El mecanismo permite al decoder COPIAR tokens directamente del
encoder input (problema), crucial para reproducir números exactos
en las respuestas matemáticas.

Arquitectura:
    1. p_gen gate: sigmoid([decoder_state, context_vector, decoder_emb])
       → probabilidad de GENERAR desde vocabulario vs COPIAR del input
    2. Copy distribution: cross-attention weights mapeadas al espacio del vocab
    3. Final: P(w) = p_gen * P_vocab(w) + (1 - p_gen) * P_copy(w)

Uso:
    from transformer_math_physics_tutor.models.transformer_v4 import TransformerV4

    config = TransformerConfig(vocab_size=4000)
    model = TransformerV4(config, answer_scale=1000.0)
    final_probs, answer_pred, p_gen = model((enc_input, dec_input), training=True)
"""

import tensorflow as tf
from transformer_math_physics_tutor.models.transformer_v3 import TransformerV3
from transformer_math_physics_tutor.models.config import TransformerConfig


class TransformerV4(TransformerV3):
    """
    Transformer v4 con Pointer-Generator Network.

    Hereda toda la arquitectura de TransformerV3 (encoder, decoder,
    final_layer, answer_head) y añade:
    - p_gen gate: decide generar vs copiar en cada timestep
    - copy distribution: atención sobre tokens del encoder input
    - blended output: mezcla de vocab probs y copy probs

    El mecanismo de copia permite al modelo reproducir exactamente
    los números del problema en la respuesta, resolviendo el problema
    de exact match = 3% que tiene v3.

    Attributes:
        p_gen_linear: Capa Dense que computa p_gen desde [state, context, emb].
        num_copy_layers: Número de capas de decoder cuyas cross-attention
                         weights se promedian para la copy distribution.
    """

    def __init__(
        self,
        config: TransformerConfig,
        answer_scale: float = 1000.0,
        num_copy_layers: int = 2,
        **kwargs
    ):
        """
        Inicializa TransformerV4.

        Args:
            config: Configuración del Transformer.
            answer_scale: Divisor para normalizar answer_value.
            num_copy_layers: Cuántas capas de decoder (desde la última)
                            usar para promediar cross-attention weights.
                            Default=2 (últimas 2 capas).
        """
        super().__init__(config, answer_scale, **kwargs)

        self.num_copy_layers = min(num_copy_layers, config.num_layers)

        # p_gen gate: [decoder_state, context_vector, decoder_emb] → sigmoid → scalar
        # Input dim: d_model * 3 (concatenación de 3 representaciones)
        self.p_gen_linear = tf.keras.layers.Dense(
            1, activation='sigmoid', name='p_gen_gate',
            bias_initializer=tf.keras.initializers.Constant(1.0)
            # Bias inicializado a 1.0 → sigmoid(1.0) ≈ 0.73
            # Sesgo inicial hacia GENERAR (no copiar) para estabilidad
        )

    def call(
        self,
        inputs: tuple,
        training: bool = False,
        return_attention: bool = False
    ):
        """
        Forward pass con Pointer-Generator.

        Args:
            inputs: Tupla de (inp, tar).
            training: Si True, aplica dropout.
            return_attention: Si True, retorna attention_weights y p_gen.

        Returns:
            Si return_attention=False:
                Tupla de (final_probs, answer_pred, p_gen_mean).
                - final_probs: shape (batch, tar_seq_len, vocab_size) — PROBABILIDADES
                - answer_pred: shape (batch,) — valor numérico predicho
                - p_gen_mean: shape () — p_gen promedio (para monitoreo)
            Si return_attention=True:
                Tupla de (final_probs, answer_pred, attention_weights, p_gen).
                - p_gen: shape (batch, tar_seq_len, 1) — p_gen por posición
        """
        inp, tar = inputs

        # --- Crear máscaras ---
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        # --- Encoder ---
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)
        # enc_output: (batch, inp_seq_len, d_model)

        # --- Decoder (siempre con attention weights para copy) ---
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training=training,
            look_ahead_mask=combined_mask, padding_mask=dec_padding_mask
        )
        # dec_output: (batch, tar_seq_len, d_model)

        # --- Vocab logits → probs ---
        vocab_logits = self.final_layer(dec_output)  # (batch, tar_seq, vocab_size)
        vocab_probs = tf.nn.softmax(vocab_logits, axis=-1)
        # vocab_probs: (batch, tar_seq, vocab_size)

        # --- Cross-attention weights para copy distribution ---
        # Promediar las últimas N capas de cross-attention
        cross_attn_weights = []
        for i in range(self.config.num_layers - self.num_copy_layers + 1,
                       self.config.num_layers + 1):
            key = f'decoder_layer{i}_block2'
            if key in attention_weights:
                cross_attn_weights.append(attention_weights[key])

        # Cada weight: (batch, num_heads, tar_seq, inp_seq)
        # Promediar sobre heads y capas
        if cross_attn_weights:
            # Stack y promediar: (N_layers, batch, heads, tar, inp) → promedio
            stacked = tf.stack(cross_attn_weights, axis=0)
            cross_attn = tf.reduce_mean(stacked, axis=[0, 2])
            # cross_attn: (batch, tar_seq, inp_seq) — distribución sobre encoder tokens
        else:
            # Fallback: última capa, promedio sobre heads
            last_key = f'decoder_layer{self.config.num_layers}_block2'
            cross_attn = tf.reduce_mean(attention_weights[last_key], axis=1)

        # --- Context vector (weighted sum of encoder output) ---
        context = tf.matmul(cross_attn, enc_output)
        # context: (batch, tar_seq, d_model)

        # --- Decoder input embedding (para p_gen gate) ---
        dec_emb = self.decoder.embedding(tar)  # (batch, tar_seq, d_model)
        dec_emb *= tf.math.sqrt(tf.cast(self.decoder.d_model, tf.float32))

        # --- p_gen gate ---
        # Concatenar: [decoder_state, context, decoder_embedding]
        p_gen_input = tf.concat([dec_output, context, dec_emb], axis=-1)
        # p_gen_input: (batch, tar_seq, 3 * d_model)
        p_gen = self.p_gen_linear(p_gen_input)
        # p_gen: (batch, tar_seq, 1) — probabilidad de GENERAR

        # --- Copy distribution: mapear attention weights al espacio del vocab ---
        # inp: (batch, inp_seq) — token IDs del encoder input
        # cross_attn: (batch, tar_seq, inp_seq) — attention weights
        # Queremos: copy_dist[batch, tar, token_id] = sum(cross_attn[batch, tar, pos]
        #           para todo pos donde inp[batch, pos] == token_id)
        inp_one_hot = tf.one_hot(inp, self.config.vocab_size)
        # inp_one_hot: (batch, inp_seq, vocab_size)
        copy_dist = tf.matmul(cross_attn, inp_one_hot)
        # copy_dist: (batch, tar_seq, vocab_size)

        # --- Blend: P(w) = p_gen * P_vocab(w) + (1 - p_gen) * P_copy(w) ---
        final_probs = p_gen * vocab_probs + (1.0 - p_gen) * copy_dist
        # final_probs: (batch, tar_seq, vocab_size) — distribución final

        # Protección numérica: asegurar que sume a ~1 y no sea 0
        final_probs = tf.maximum(final_probs, 1e-9)

        # --- Answer head (heredado de TransformerV3) ---
        enc_mask = tf.cast(tf.not_equal(inp, 0), tf.float32)
        enc_mask_expanded = enc_mask[:, :, tf.newaxis]
        pooled = tf.reduce_sum(enc_output * enc_mask_expanded, axis=1)
        n_real_enc = tf.maximum(tf.reduce_sum(enc_mask, axis=1, keepdims=True), 1.0)
        pooled = pooled / n_real_enc

        answer_hidden = self.answer_dense1(pooled)
        answer_hidden = self.answer_dropout(answer_hidden, training=training)
        answer_pred = self.answer_dense2(answer_hidden)
        answer_pred = tf.squeeze(answer_pred, axis=-1)

        # --- p_gen promedio (para monitoreo) ---
        # Enmascarar posiciones de padding del target
        tar_mask = tf.cast(tf.not_equal(tar, 0), tf.float32)  # (batch, tar_seq)
        tar_mask_expanded = tar_mask[:, :, tf.newaxis]  # (batch, tar_seq, 1)
        p_gen_masked = p_gen * tar_mask_expanded
        p_gen_mean = (tf.reduce_sum(p_gen_masked) /
                      tf.maximum(tf.reduce_sum(tar_mask), 1.0))

        if return_attention:
            return final_probs, answer_pred, attention_weights, p_gen
        return final_probs, answer_pred, p_gen_mean


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: TransformerV4 (Pointer-Generator)")
    print("=" * 60)

    config = TransformerConfig(
        d_model=64, num_heads=4, num_layers=2,
        dff=128, vocab_size=100,
        max_encoder_len=20, max_decoder_len=15
    )

    model = TransformerV4(config, answer_scale=1000.0)

    # Test inputs
    enc_inp = tf.random.uniform((2, 20), 1, 100, dtype=tf.int32)
    dec_inp = tf.random.uniform((2, 15), 1, 100, dtype=tf.int32)

    # Forward pass sin return_attention
    final_probs, answer_pred, p_gen_mean = model(
        (enc_inp, dec_inp), training=False
    )
    print(f"Encoder input: {enc_inp.shape}")
    print(f"Decoder input: {dec_inp.shape}")
    print(f"Final probs: {final_probs.shape}")  # (2, 15, 100)
    print(f"Answer pred: {answer_pred.shape}")   # (2,)
    print(f"p_gen mean: {p_gen_mean.numpy():.4f}")
    print(f"Probs sum check: {tf.reduce_sum(final_probs[0, 0, :]).numpy():.4f}")

    # Forward pass con return_attention
    final_probs, answer_pred, attn_weights, p_gen = model(
        (enc_inp, dec_inp), training=False, return_attention=True
    )
    print(f"\nWith return_attention:")
    print(f"p_gen shape: {p_gen.shape}")  # (2, 15, 1)
    print(f"p_gen[0,0]: {p_gen[0, 0, 0].numpy():.4f}")
    print(f"Attention weight keys: {list(attn_weights.keys())}")

    # Parámetros
    total_params = model.count_params()
    print(f"\nTotal params: {total_params:,}")
