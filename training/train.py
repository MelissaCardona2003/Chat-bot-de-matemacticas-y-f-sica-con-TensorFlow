"""
train.py — Loop de entrenamiento con GradientTape.

Implementa el pipeline de entrenamiento completo del Transformer,
incluyendo custom training loop, evaluación y checkpointing.

Uso:
    from transformer_math_physics_tutor.training.train import TransformerTrainer
    from transformer_math_physics_tutor.models.transformer import Transformer
    from transformer_math_physics_tutor.models.config import TransformerConfig

    config = TransformerConfig(vocab_size=128)
    model = Transformer(config)
    trainer = TransformerTrainer(model, config)
    trainer.train(train_dataset, val_dataset, epochs=50)
"""

import os
import time
import json
import tensorflow as tf
from pathlib import Path
from typing import Optional

from transformer_math_physics_tutor.models.config import TransformerConfig
from transformer_math_physics_tutor.training.scheduler import CustomSchedule
from transformer_math_physics_tutor.training.losses import (
    loss_function, accuracy_function, cross_attention_entropy_loss
)


BASE_DIR = Path(__file__).resolve().parent.parent


class TransformerTrainer:
    """
    Entrenador del modelo Transformer con custom training loop.

    Implementa:
    - Custom learning rate schedule (warmup + decay)
    - Adam optimizer con betas del paper original
    - Training loop con tf.GradientTape
    - Evaluación en validation set
    - Checkpointing periódico
    - Logging de métricas

    Attributes:
        model: Instancia del Transformer.
        config: Instancia de TransformerConfig.
        optimizer: Adam optimizer con custom schedule.
        train_loss: Métrica Mean para loss de entrenamiento.
        train_accuracy: Métrica Mean para accuracy de entrenamiento.
    """

    def __init__(self, model: tf.keras.Model, config: TransformerConfig,
                 decoder_mask_rate: float = 0.0,
                 attn_diversity_weight: float = 0.0):
        """
        Inicializa el trainer.

        Args:
            model: Modelo Transformer instanciado.
            config: Configuración de hiperparámetros.
            decoder_mask_rate: Fracción de tokens del decoder input a enmascarar
                durante entrenamiento (0.0 = desactivado). Valores 0.15-0.25
                fuerzan al modelo a usar cross-attention con el encoder.
            attn_diversity_weight: Peso del loss de diversidad de cross-attention.
                (0.0 = desactivado). Penaliza atención uniforme para forzar
                al modelo a focalizar en tokens relevantes del encoder.
                Valores recomendados: 0.5-2.0.
        """
        self.model = model
        self.config = config
        self.decoder_mask_rate = decoder_mask_rate
        self.attn_diversity_weight = attn_diversity_weight

        # Learning rate schedule del paper original
        self.learning_rate = CustomSchedule(config.d_model, config.warmup_steps)

        # Adam con betas del paper: β1=0.9, β2=0.98, ε=1e-9
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )

        # Métricas de seguimiento (en CPU — Blackwell workaround)
        with tf.device('/CPU:0'):
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        # Directorio de checkpoints
        self.checkpoint_dir = BASE_DIR / config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint manager
        self.ckpt = tf.train.Checkpoint(
            transformer=self.model,
            optimizer=self.optimizer
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt,
            str(self.checkpoint_dir),
            max_to_keep=5
        )

        # Historial de entrenamiento
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rates": [],
        }

    def _apply_decoder_masking(self, tar_inp: tf.Tensor) -> tf.Tensor:
        """
        Enmascara aleatoriamente tokens del decoder input durante training.

        Reemplaza una fracción de tokens con IDs aleatorios para ROMPER
        el atajo autoregresivo y forzar al modelo a usar cross-attention
        con el encoder para recuperar la información perdida.

        Protecciones:
        - NO enmascara la posición 0 (token START — necesario para iniciar)
        - NO enmascara tokens PAD (id=0)

        Args:
            tar_inp: Decoder input, shape (batch, seq_len).

        Returns:
            Decoder input con tokens enmascarados.
        """
        shape = tf.shape(tar_inp)
        batch_size, seq_len = shape[0], shape[1]

        # Máscara aleatoria: True = enmascarar este token
        random_vals = tf.random.uniform([batch_size, seq_len])
        should_mask = random_vals < self.decoder_mask_rate

        # Proteger posición 0 (START) — nunca enmascarar
        pos0_protect = tf.concat([
            tf.zeros([batch_size, 1], dtype=tf.bool),
            tf.ones([batch_size, seq_len - 1], dtype=tf.bool)
        ], axis=1)
        should_mask = tf.logical_and(should_mask, pos0_protect)

        # Proteger PAD tokens (id=0) — no enmascarar padding
        non_pad = tf.not_equal(tar_inp, 0)
        should_mask = tf.logical_and(should_mask, non_pad)

        # Tokens de reemplazo: IDs aleatorios del vocabulario (4..vocab-1)
        # Evitamos special tokens (0=PAD, 1=START, 2=END, 3=UNK)
        random_tokens = tf.random.uniform(
            [batch_size, seq_len],
            minval=4,
            maxval=self.config.vocab_size,
            dtype=tf.int32
        )

        # Aplicar: donde should_mask=True, usar random_token; sino, original
        tar_inp_masked = tf.where(should_mask, random_tokens, tar_inp)
        return tar_inp_masked

    @tf.function
    def train_step(
        self,
        inp: tf.Tensor,
        tar_inp: tf.Tensor,
        tar_real: tf.Tensor
    ) -> None:
        """
        Un paso de entrenamiento con GradientTape.

        Implementa teacher forcing: el decoder recibe decoder_input
        y se compara contra decoder_target.

        Si decoder_mask_rate > 0, enmascara aleatoriamente tokens del
        decoder input para forzar al modelo a usar cross-attention.

        Si attn_diversity_weight > 0, añade un loss que penaliza
        cross-attention uniforme (alta entropía) para forzar focalización.

        Args:
            inp: Tensor de entrada del encoder, shape (batch, inp_seq_len).
            tar_inp: Decoder input (teacher forcing), shape (batch, tar_seq_len).
                     Secuencia con START al inicio, sin END al final.
            tar_real: Decoder target, shape (batch, tar_seq_len).
                      Secuencia sin START al inicio, con END al final.

        Returns:
            None (actualiza pesos y métricas in-place).
        """
        # Aplicar decoder masking si está activado
        if self.decoder_mask_rate > 0:
            tar_inp = self._apply_decoder_masking(tar_inp)

        with tf.GradientTape() as tape:
            # Forward pass — con attention weights si necesitamos diversity loss
            if self.attn_diversity_weight > 0:
                predictions, attn_weights = self.model(
                    (inp, tar_inp), training=True, return_attention=True
                )
            else:
                predictions = self.model((inp, tar_inp), training=True)

            # Calcular loss principal (cross-entropy con máscara y label smoothing)
            loss = loss_function(tar_real, predictions, self.config.label_smoothing)

            # Añadir loss de diversidad de cross-attention
            if self.attn_diversity_weight > 0:
                diversity_loss = cross_attention_entropy_loss(attn_weights, inp)
                loss = loss + self.attn_diversity_weight * diversity_loss

        # Calcular gradientes
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Gradient clipping para estabilidad
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        # Aplicar gradientes y actualizar métricas en CPU
        # (Blackwell workaround — variables en CPU, XLA compila forward/backward en GPU)
        with tf.device('/CPU:0'):
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            # Actualizar métricas
            self.train_loss(loss)
            self.train_accuracy(accuracy_function(tar_real, predictions))

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: Optional[int] = None,
        checkpoint_every: int = 5,
        verbose: bool = True,
        early_stopping_patience: int = 0
    ) -> dict:
        """
        Loop principal de entrenamiento.

        Args:
            train_dataset: Dataset de entrenamiento con estructura
                          ((encoder_input, decoder_input), decoder_target).
            val_dataset: Dataset de validación (opcional).
            epochs: Número de épocas. Si None, usa config.epochs.
            checkpoint_every: Guardar checkpoint cada N épocas.
            verbose: Si True, imprime progreso.
            early_stopping_patience: Número de épocas sin mejora en val_loss
                                     antes de detener. 0 = desactivado.

        Returns:
            Diccionario con historial de métricas.
        """
        if epochs is None:
            epochs = self.config.epochs

        # Restaurar último checkpoint si existe
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            if verbose:
                print(f"Checkpoint restaurado: {self.ckpt_manager.latest_checkpoint}")

        if verbose:
            print(f"\nIniciando entrenamiento por {epochs} épocas...")
            print(f"Dispositivo: {self._get_device()}")
            if early_stopping_patience > 0:
                print(f"Early stopping: patience={early_stopping_patience}")
            print("-" * 60)

        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(epochs):
            start = time.time()

            # Resetear métricas al inicio de cada época
            with tf.device('/CPU:0'):
                self.train_loss.reset_state()
                self.train_accuracy.reset_state()

            # Iterar sobre batches
            for batch_idx, ((inp, dec_inp), dec_target) in enumerate(train_dataset):
                # inp: encoder_input (problema tokenizado)
                # dec_inp: decoder_input [START, tok1, ..., PAD]
                # dec_target: decoder_target [tok1, ..., END, PAD]
                self.train_step(inp, dec_inp, dec_target)

                if verbose and (batch_idx + 1) % 50 == 0:
                    with tf.device('/CPU:0'):
                        _loss = self.train_loss.result()
                        _acc = self.train_accuracy.result()
                    print(f"  Época {epoch+1}, Batch {batch_idx+1}: "
                          f"Loss={_loss:.4f}, "
                          f"Acc={_acc:.4f}")

            # Métricas de la época
            with tf.device('/CPU:0'):
                epoch_loss = self.train_loss.result().numpy()
                epoch_acc = self.train_accuracy.result().numpy()
            epoch_time = time.time() - start

            self.history["train_loss"].append(float(epoch_loss))
            self.history["train_accuracy"].append(float(epoch_acc))

            # Obtener learning rate actual
            with tf.device('/CPU:0'):
                current_lr = float(self.learning_rate(self.optimizer.iterations))
            self.history["learning_rates"].append(current_lr)

            # Evaluación en validation
            if val_dataset is not None:
                val_loss, val_acc = self.evaluate(val_dataset)
                self.history["val_loss"].append(float(val_loss))
                self.history["val_accuracy"].append(float(val_acc))
            else:
                val_loss = val_acc = None

            # Logging
            if verbose:
                msg = (f"Época {epoch+1}/{epochs} — "
                       f"Loss: {epoch_loss:.4f} — "
                       f"Acc: {epoch_acc:.4f}")
                if val_loss is not None:
                    msg += f" — Val_Loss: {val_loss:.4f} — Val_Acc: {val_acc:.4f}"
                msg += f" — LR: {current_lr:.6f} — Tiempo: {epoch_time:.1f}s"
                print(msg)

            # Guardar checkpoint
            if (epoch + 1) % checkpoint_every == 0:
                ckpt_path = self.ckpt_manager.save()
                if verbose:
                    print(f"  → Checkpoint guardado: {ckpt_path}")

            # Early stopping
            if early_stopping_patience > 0 and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    # Guardar mejor modelo (pesos .h5) cuando mejora val_loss
                    best_weights_path = str(self.checkpoint_dir / "best_model.weights.h5")
                    self.model.save_weights(best_weights_path)
                    ckpt_path = self.ckpt_manager.save()
                    if verbose:
                        print(f"  → ✅ Mejor val_loss: {val_loss:.4f} — "
                              f"modelo guardado (época {best_epoch})")
                else:
                    patience_counter += 1
                    if verbose:
                        print(f"  → Sin mejora ({patience_counter}/{early_stopping_patience}) "
                              f"— mejor: {best_val_loss:.4f} en época {best_epoch}")
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"\n  🛑 Early stopping en época {epoch+1}")
                            print(f"     Mejor val_loss: {best_val_loss:.4f} en época {best_epoch}")
                            print(f"     Restaurando mejores pesos...")
                        # Restaurar los mejores pesos
                        best_weights_path = str(self.checkpoint_dir / "best_model.weights.h5")
                        if os.path.exists(best_weights_path):
                            self.model.load_weights(best_weights_path)
                        break

        # Guardar historial
        self._save_history()

        if verbose:
            print("-" * 60)
            print("Entrenamiento completado!")

        return self.history

    def evaluate(self, dataset: tf.data.Dataset) -> tuple:
        """
        Evalúa el modelo en un dataset (loss y accuracy).

        Args:
            dataset: Dataset con estructura ((enc_inp, dec_inp), target).

        Returns:
            Tupla de (loss_promedio, accuracy_promedio).
        """
        with tf.device('/CPU:0'):
            val_loss = tf.keras.metrics.Mean()
            val_accuracy = tf.keras.metrics.Mean()

        @tf.function
        def eval_step(inp, dec_inp, dec_target):
            predictions = self.model((inp, dec_inp), training=False)
            loss = loss_function(dec_target, predictions, self.config.label_smoothing)
            acc = accuracy_function(dec_target, predictions)
            return loss, acc

        for (inp, dec_inp), dec_target in dataset:
            loss, acc = eval_step(inp, dec_inp, dec_target)
            with tf.device('/CPU:0'):
                val_loss(loss)
                val_accuracy(acc)

        with tf.device('/CPU:0'):
            return val_loss.result().numpy(), val_accuracy.result().numpy()

    def save_model(self, filepath: Optional[str] = None) -> None:
        """
        Guarda los pesos del modelo.

        Args:
            filepath: Ruta para guardar. Si None, usa checkpoint_dir/model_weights.
        """
        if filepath is None:
            filepath = str(self.checkpoint_dir / "model_weights.weights.h5")

        self.model.save_weights(filepath)
        print(f"Pesos del modelo guardados en {filepath}")

    def load_model(self, filepath: Optional[str] = None) -> None:
        """
        Carga los pesos del modelo.

        Args:
            filepath: Ruta de los pesos. Si None, restaura último checkpoint.
        """
        if filepath is not None:
            self.model.load_weights(filepath)
            print(f"Pesos cargados desde {filepath}")
        elif self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"Checkpoint restaurado: {self.ckpt_manager.latest_checkpoint}")
        else:
            print("No se encontraron pesos ni checkpoints para cargar.")

    def _save_history(self) -> None:
        """Guarda el historial de entrenamiento en JSON."""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    @staticmethod
    def _get_device() -> str:
        """Detecta si hay GPU disponible."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return f"GPU ({len(gpus)} dispositivo(s))"
        return "CPU"


if __name__ == "__main__":
    from transformer_math_physics_tutor.models.transformer import Transformer

    print("=" * 60)
    print("DEMO: TransformerTrainer")
    print("=" * 60)

    # Crear configuración y modelo
    config = TransformerConfig(
        d_model=64,
        num_heads=4,
        num_layers=2,
        dff=128,
        vocab_size=50,
        epochs=3,
        batch_size=4
    )

    model = Transformer(config)
    trainer = TransformerTrainer(model, config)

    # Crear dataset dummy
    n_samples = 20
    enc_inp = tf.random.uniform((n_samples, 20), 1, 50, dtype=tf.int32)
    dec_inp = tf.random.uniform((n_samples, 15), 1, 50, dtype=tf.int32)
    targets = tf.random.uniform((n_samples, 15), 1, 50, dtype=tf.int32)

    dummy_dataset = tf.data.Dataset.from_tensor_slices(
        ((enc_inp, dec_inp), targets)
    ).batch(4)

    # Entrenar
    history = trainer.train(dummy_dataset, epochs=3)

    print(f"\nHistorial: {list(history.keys())}")
    print(f"Loss final: {history['train_loss'][-1]:.4f}")
    print(f"Accuracy final: {history['train_accuracy'][-1]:.4f}")
