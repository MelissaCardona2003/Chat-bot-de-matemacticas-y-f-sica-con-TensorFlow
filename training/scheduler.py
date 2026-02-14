"""
scheduler.py — Learning Rate Scheduler con Warmup.

Implementa el schedule del paper "Attention Is All You Need":
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

El learning rate crece linealmente durante warmup_steps,
luego decrece proporcionalmente a la raíz cuadrada inversa del paso.

Uso:
    from transformer_math_physics_tutor.training.scheduler import CustomSchedule
    lr_schedule = CustomSchedule(d_model=256, warmup_steps=4000)
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98)
"""

import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning Rate Schedule del paper original del Transformer.

    Comportamiento:
    - Fase de warmup (0 → warmup_steps): LR crece linealmente
    - Fase de decay (warmup_steps → ∞): LR decrece como step^(-0.5)

    Esto permite:
    1. Empezar con LR pequeño para estabilizar el entrenamiento inicial
    2. Aumentar gradualmente hasta un pico
    3. Luego decrecer para convergencia fina

    Attributes:
        d_model: Dimensión del modelo (factor de escala).
        warmup_steps: Número de pasos de warmup.
    """

    def __init__(self, d_model: int, warmup_steps: int = 4000):
        """
        Inicializa el scheduler.

        Args:
            d_model: Dimensión del modelo. El LR máximo escala con d_model^(-0.5).
            warmup_steps: Número de pasos de calentamiento.
                         Mayor warmup = LR máximo más bajo pero más estable.
        """
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model_float = tf.constant(float(d_model), dtype=tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        """
        Calcula el learning rate para un paso dado.

        Fórmula: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

        Args:
            step: Paso actual de entrenamiento.

        Returns:
            Learning rate para el paso actual.
        """
        step = tf.cast(step, tf.float32)

        # Evitar división por cero en el primer paso
        step = tf.maximum(step, 1.0)

        # Argumento 1: decaimiento del LR (para steps > warmup)
        arg1 = tf.math.rsqrt(step)  # step^(-0.5)

        # Argumento 2: crecimiento lineal del LR (para steps < warmup)
        arg2 = step * (self.warmup_steps ** -1.5)

        # Tomar el mínimo de ambos (warmup crece, luego decay domina)
        lr = tf.math.rsqrt(self.d_model_float) * tf.math.minimum(arg1, arg2)

        return lr

    def get_config(self):
        """Serialización para guardar/cargar el scheduler."""
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Custom Learning Rate Schedule")
    print("=" * 60)

    # Crear scheduler
    schedule = CustomSchedule(d_model=256, warmup_steps=4000)

    # Calcular LR para diferentes pasos
    steps = [1, 100, 500, 1000, 2000, 4000, 8000, 20000, 50000]
    for step in steps:
        lr = schedule(step).numpy()
        print(f"  Step {step:>6d}: lr = {lr:.6f}")

    # LR máximo (ocurre alrededor del paso warmup)
    peak_lr = schedule(4000).numpy()
    print(f"\nLR pico (en warmup_steps=4000): {peak_lr:.6f}")
