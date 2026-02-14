"""
transformer.py — Modelo Transformer completo (Encoder-Decoder).

Ensambla todas las piezas: Encoder, Decoder y capa densa final.
Incluye creación de máscaras (padding y look-ahead).

Uso:
    from transformer_math_physics_tutor.models.transformer import Transformer
    from transformer_math_physics_tutor.models.config import TransformerConfig

    config = TransformerConfig(vocab_size=128)
    model = Transformer(config)
    output = model((encoder_input, decoder_input), training=True)
"""

import tensorflow as tf
from transformer_math_physics_tutor.models.positional_encoding import positional_encoding
from transformer_math_physics_tutor.models.encoder_layer import EncoderLayer
from transformer_math_physics_tutor.models.decoder_layer import DecoderLayer
from transformer_math_physics_tutor.models.config import TransformerConfig
from transformer_math_physics_tutor.models.xla_dropout import XLADropout


class Encoder(tf.keras.layers.Layer):
    """
    Stack de capas de Encoder del Transformer.

    Flujo:
        token_ids → Embedding → * sqrt(d_model) → + Positional Encoding
        → Dropout → EncoderLayer_1 → EncoderLayer_2 → ... → output

    La multiplicación por sqrt(d_model) es para escalar los embeddings
    antes de sumar el positional encoding (ambos deben tener magnitudes
    comparables).

    Attributes:
        d_model: Dimensión del modelo.
        num_layers: Número de capas de encoder.
        embedding: Capa de embedding.
        pos_encoding: Codificación posicional pre-calculada.
        enc_layers: Lista de capas EncoderLayer.
        dropout: Dropout tras embedding + positional.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        input_vocab_size: int,
        maximum_position_encoding: int,
        rate: float = 0.1,
        **kwargs
    ):
        """
        Inicializa el Encoder.

        Args:
            num_layers: Número de capas de encoder.
            d_model: Dimensión del modelo.
            num_heads: Número de cabezas de atención.
            dff: Dimensión de la FFN.
            input_vocab_size: Tamaño del vocabulario de entrada.
            maximum_position_encoding: Longitud máxima de secuencia.
            rate: Tasa de dropout.
        """
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.supports_masking = True

        # Capa de embedding: token_id → vector de d_model dimensiones
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        # Positional encoding pre-calculado (no entrenable)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        # Stack de capas de encoder
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.dropout = XLADropout(rate)

    def call(
        self,
        x: tf.Tensor,
        training: bool = False,
        mask: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Forward pass del Encoder.

        Args:
            x: Tensor de token IDs, shape (batch_size, seq_len).
            training: Si True, aplica dropout.
            mask: Máscara de padding.

        Returns:
            Tensor de shape (batch_size, seq_len, d_model).
        """
        seq_len = tf.shape(x)[1]

        # Embedding + escalado
        # El escalado por sqrt(d_model) es del paper original
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Sumar positional encoding
        # Recortar pos_encoding a la longitud actual de la secuencia
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # Pasar por cada capa de encoder
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training, mask=mask)

        return x  # (batch, seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """
    Stack de capas de Decoder del Transformer.

    Similar al Encoder, pero cada capa tiene además
    cross-attention con la salida del encoder.

    Attributes:
        d_model: Dimensión del modelo.
        num_layers: Número de capas de decoder.
        embedding: Capa de embedding.
        pos_encoding: Codificación posicional pre-calculada.
        dec_layers: Lista de capas DecoderLayer.
        dropout: Dropout tras embedding + positional.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        target_vocab_size: int,
        maximum_position_encoding: int,
        rate: float = 0.1,
        **kwargs
    ):
        """
        Inicializa el Decoder.

        Args:
            num_layers: Número de capas de decoder.
            d_model: Dimensión del modelo.
            num_heads: Número de cabezas de atención.
            dff: Dimensión de la FFN.
            target_vocab_size: Tamaño del vocabulario de salida.
            maximum_position_encoding: Longitud máxima de secuencia.
            rate: Tasa de dropout.
        """
        super(Decoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.dropout = XLADropout(rate)

    def call(
        self,
        x: tf.Tensor,
        enc_output: tf.Tensor,
        training: bool = False,
        look_ahead_mask: tf.Tensor = None,
        padding_mask: tf.Tensor = None
    ) -> tuple:
        """
        Forward pass del Decoder.

        Args:
            x: Token IDs del target, shape (batch, target_seq_len).
            enc_output: Salida del Encoder, shape (batch, input_seq_len, d_model).
            training: Si True, aplica dropout.
            look_ahead_mask: Máscara triangular para self-attention.
            padding_mask: Máscara de padding para cross-attention.

        Returns:
            Tupla de (x, attention_weights):
            - x: shape (batch, target_seq_len, d_model)
            - attention_weights: dict con pesos de atención por capa
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Embedding + escalado + positional encoding
        x = self.embedding(x)  # (batch, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        # Pasar por cada capa de decoder
        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(
                x, enc_output, training=training,
                look_ahead_mask=look_ahead_mask, padding_mask=padding_mask
            )
            # Guardar pesos de atención para visualización
            attention_weights[f'decoder_layer{i+1}_block1'] = block1  # Self-attn
            attention_weights[f'decoder_layer{i+1}_block2'] = block2  # Cross-attn

        return x, attention_weights


class Transformer(tf.keras.Model):
    """
    Modelo Transformer completo (Encoder-Decoder).

    Arquitectura:
        Encoder: input_tokens → encoder_output
        Decoder: target_tokens + encoder_output → decoder_output
        Linear: decoder_output → logits (sobre vocabulario)

    El modelo crea automáticamente las máscaras necesarias
    (padding mask y look-ahead mask) a partir de las secuencias de entrada.

    Attributes:
        encoder: Stack de capas de encoder.
        decoder: Stack de capas de decoder.
        final_layer: Capa Dense final que proyecta a vocab_size.
    """

    def __init__(self, config: TransformerConfig, **kwargs):
        """
        Inicializa el Transformer.

        Args:
            config: Instancia de TransformerConfig con todos los hiperparámetros.
        """
        super(Transformer, self).__init__(**kwargs)

        self.config = config

        self.encoder = Encoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            dff=config.dff,
            input_vocab_size=config.vocab_size,
            maximum_position_encoding=config.max_encoder_len,
            rate=config.dropout_rate
        )

        self.decoder = Decoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            dff=config.dff,
            target_vocab_size=config.vocab_size,
            maximum_position_encoding=config.max_decoder_len,
            rate=config.dropout_rate
        )

        # Capa final: proyecta d_model → vocab_size (logits)
        self.final_layer = tf.keras.layers.Dense(config.vocab_size)

    @staticmethod
    def create_padding_mask(seq: tf.Tensor) -> tf.Tensor:
        """
        Crea máscara de padding (para ignorar tokens <PAD> = 0).

        La máscara tiene 1.0 donde hay padding y 0.0 donde hay tokens reales.
        Se usa en attention para que los tokens <PAD> no reciban atención.

        Args:
            seq: Tensor de token IDs, shape (batch, seq_len).

        Returns:
            Tensor de máscara, shape (batch, 1, 1, seq_len).
            Los ejes extra son para broadcasting con attention weights.
        """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # Añadir dimensiones para broadcasting con (batch, heads, seq_q, seq_k)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size: int) -> tf.Tensor:
        """
        Crea máscara look-ahead (triangular superior).

        Previene que el decoder vea tokens futuros en self-attention.
        La máscara tiene 1.0 en posiciones "futuras" (que deben ser bloqueadas).

        Ejemplo para size=4:
            [[0, 1, 1, 1],
             [0, 0, 1, 1],
             [0, 0, 0, 1],
             [0, 0, 0, 0]]

        Args:
            size: Tamaño de la secuencia (crea máscara size x size).

        Returns:
            Tensor de máscara, shape (size, size).
        """
        # band_part con lower=-1 (todos), upper=0 (ninguno superior) = triangular inferior
        # 1 - triangular_inferior = triangular superior (lo que queremos bloquear)
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (size, size)

    def call(
        self,
        inputs: tuple,
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass del Transformer completo.

        Args:
            inputs: Tupla de (inp, tar):
                - inp: Token IDs del encoder, shape (batch, inp_seq_len).
                - tar: Token IDs del decoder, shape (batch, tar_seq_len).
            training: Si True, aplica dropout.

        Returns:
            Tensor de logits, shape (batch, tar_seq_len, vocab_size).
        """
        inp, tar = inputs

        # --- Crear todas las máscaras ---

        # 1. Padding mask para el encoder (ignora <PAD> en input)
        enc_padding_mask = self.create_padding_mask(inp)
        # shape: (batch, 1, 1, inp_seq_len)

        # 2. Padding mask para cross-attention en decoder (ignora <PAD> en input)
        dec_padding_mask = self.create_padding_mask(inp)
        # shape: (batch, 1, 1, inp_seq_len)

        # 3. Look-ahead mask para self-attention en decoder
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        # shape: (tar_seq_len, tar_seq_len)

        # 4. Padding mask para el target (ignora <PAD> en target)
        dec_target_padding_mask = self.create_padding_mask(tar)
        # shape: (batch, 1, 1, tar_seq_len)

        # 5. Máscara combinada: máximo de look_ahead y padding del target
        # Esto bloquea tanto tokens futuros como tokens de padding
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        # shape: (batch, 1, tar_seq_len, tar_seq_len) por broadcasting

        # --- Encoder ---
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)
        # shape: (batch, inp_seq_len, d_model)

        # --- Decoder ---
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training=training,
            look_ahead_mask=combined_mask, padding_mask=dec_padding_mask
        )
        # dec_output shape: (batch, tar_seq_len, d_model)

        # --- Capa lineal final ---
        final_output = self.final_layer(dec_output)
        # shape: (batch, tar_seq_len, vocab_size)

        return final_output

    def get_config(self):
        """Serialización de la configuración."""
        config = super().get_config()
        config.update(self.config.to_dict())
        return config


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Transformer Completo")
    print("=" * 60)

    # Configuración
    config = TransformerConfig(
        d_model=256,
        num_heads=8,
        num_layers=4,
        dff=1024,
        dropout_rate=0.1,
        max_encoder_len=100,
        max_decoder_len=150,
        vocab_size=128
    )

    # Crear modelo
    model = Transformer(config)

    # Entrada de prueba
    # Simular batch de 2 problemas
    encoder_input = tf.random.uniform((2, 20), minval=0, maxval=128, dtype=tf.int32)
    decoder_input = tf.random.uniform((2, 15), minval=0, maxval=128, dtype=tf.int32)

    # Forward pass
    output = model((encoder_input, decoder_input), training=False)

    print(f"Encoder input shape: {encoder_input.shape}")   # (2, 20)
    print(f"Decoder input shape: {decoder_input.shape}")   # (2, 15)
    print(f"Output shape: {output.shape}")                  # (2, 15, 128)
    print(f"Output representa logits sobre {config.vocab_size} tokens")

    # Resumen del modelo
    model.summary()
    print(f"\nTotal de parámetros: {model.count_params():,}")
