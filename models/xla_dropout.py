"""
xla_dropout.py — Dropout compatible con XLA para GPUs Blackwell.

Keras 3's Dropout usa un SeedGenerator con estado que genera operaciones
de actualización de variables fuera del cluster XLA. Esto causa errores
con GPUs que requieren XLA (como RTX 5060 Blackwell, sm_120 + TF 2.20).

Esta implementación usa tf.random.uniform (stateless dentro de XLA) en
vez del SeedGenerator de Keras, permitiendo que el auto-jit compile
todo el grafo correctamente.

Uso:
    from transformer_math_physics_tutor.models.xla_dropout import XLADropout
    dropout = XLADropout(rate=0.1)
    output = dropout(x, training=True)
"""

import tensorflow as tf


class XLADropout(tf.keras.layers.Layer):
    """
    Dropout layer compatible con XLA auto-jit.

    No usa SeedGenerator de Keras — usa tf.random.uniform que puede
    compilarse completamente dentro de un cluster XLA.
    """

    def __init__(self, rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if not training or self.rate == 0.0:
            return inputs
        keep_prob = 1.0 - self.rate
        mask = tf.random.uniform(tf.shape(inputs)) < keep_prob
        return inputs * tf.cast(mask, inputs.dtype) / keep_prob

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config
