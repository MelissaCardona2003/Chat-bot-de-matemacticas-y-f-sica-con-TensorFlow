"""
trainer.py — Entrenador con loss combinado (seq2seq + answer regression).

Maneja:
1. Loss seq2seq (cross-entropy con máscara)
2. Loss de regresión numérica (Huber loss vía answer head)
3. Loss de diversidad de cross-attention (entropía)
4. Decoder denoising/masking (35%)

loss_total = loss_seq2seq + λ_answer * loss_answer + λ_attn * loss_attn_diversity

Uso:
    from transformer_math_physics_tutor.training.trainer import TransformerTrainer
    trainer = TransformerTrainer(model, config, answer_weight=0.5)
    trainer.train(train_dataset, val_dataset, epochs=100)
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
from transformer_math_physics_tutor.models.transformer_v3 import answer_regression_loss


BASE_DIR = Path(__file__).resolve().parent.parent


class TransformerTrainerV3:
    """
    Entrenador del Transformer v3 con loss combinado.

    A diferencia de TransformerTrainer (v1/v2), este trainer:
    - Espera datasets con estructura ((enc, dec), (target, answer_value))
    - Calcula loss combinado: seq2seq + answer + diversity
    - Trackea métricas adicionales: answer_loss, answer_mae

    Attributes:
        model: TransformerV3 con answer head.
        config: TransformerConfig.
        answer_weight: Peso λ para answer regression loss.
        attn_diversity_weight: Peso μ para diversity loss.
        decoder_mask_rate: Fracción de tokens a enmascarar.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        config: TransformerConfig,
        answer_weight: float = 0.5,
        answer_scale: float = 1000.0,
        decoder_mask_rate: float = 0.35,
        attn_diversity_weight: float = 1.0,
    ):
        self.model = model
        self.config = config
        self.answer_weight = answer_weight
        self.answer_scale = answer_scale
        self.decoder_mask_rate = decoder_mask_rate
        self.attn_diversity_weight = attn_diversity_weight

        # Learning rate schedule
        self.learning_rate = CustomSchedule(config.d_model, config.warmup_steps,
                                            scale=getattr(config, 'lr_scale', 1.0))

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )

        # Métricas (en CPU — Blackwell workaround)
        with tf.device('/CPU:0'):
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
            self.train_answer_loss = tf.keras.metrics.Mean(name='train_answer_loss')
            self.train_answer_mae = tf.keras.metrics.Mean(name='train_answer_mae')

        # Checkpoint
        self.checkpoint_dir = BASE_DIR / config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt = tf.train.Checkpoint(
            transformer=self.model,
            optimizer=self.optimizer
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt,
            str(self.checkpoint_dir),
            max_to_keep=5
        )

        # Historial
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_answer_loss": [],
            "train_answer_mae": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_answer_loss": [],
            "val_answer_mae": [],
            "learning_rates": [],
        }

    def _apply_decoder_masking(self, tar_inp: tf.Tensor) -> tf.Tensor:
        """
        Enmascara aleatoriamente tokens del decoder input.

        35% de tokens reemplazados por IDs aleatorios → fuerza al
        modelo a depender de cross-attention con el encoder.

        Protecciones:
        - NO enmascara START (posición 0)
        - NO enmascara PAD (id=0)
        """
        shape = tf.shape(tar_inp)
        batch_size, seq_len = shape[0], shape[1]

        random_vals = tf.random.uniform([batch_size, seq_len])
        should_mask = random_vals < self.decoder_mask_rate

        # Proteger posición 0 (START)
        pos0_protect = tf.concat([
            tf.zeros([batch_size, 1], dtype=tf.bool),
            tf.ones([batch_size, seq_len - 1], dtype=tf.bool)
        ], axis=1)
        should_mask = tf.logical_and(should_mask, pos0_protect)

        # Proteger PAD
        non_pad = tf.not_equal(tar_inp, 0)
        should_mask = tf.logical_and(should_mask, non_pad)

        # Tokens de reemplazo aleatorios (evitar special tokens 0-3)
        random_tokens = tf.random.uniform(
            [batch_size, seq_len],
            minval=4,
            maxval=self.config.vocab_size,
            dtype=tf.int32
        )

        return tf.where(should_mask, random_tokens, tar_inp)

    @tf.function
    def train_step(
        self,
        inp: tf.Tensor,
        tar_inp: tf.Tensor,
        tar_real: tf.Tensor,
        answer_true: tf.Tensor,
    ) -> None:
        """
        Un paso de entrenamiento con loss combinado.

        Args:
            inp: Encoder input, shape (batch, inp_seq_len).
            tar_inp: Decoder input, shape (batch, tar_seq_len).
            tar_real: Decoder target, shape (batch, tar_seq_len).
            answer_true: Valor numérico verdadero, shape (batch,).
        """
        # Aplicar decoder masking
        if self.decoder_mask_rate > 0:
            tar_inp = self._apply_decoder_masking(tar_inp)

        with tf.GradientTape() as tape:
            # Forward pass (TransformerV3 retorna logits + answer_pred + attn)
            predictions, answer_pred, attn_weights = self.model(
                (inp, tar_inp), training=True, return_attention=True
            )

            # 1. Loss seq2seq (cross-entropy con máscara)
            seq_loss = loss_function(tar_real, predictions, self.config.label_smoothing)

            # 2. Loss de regresión numérica
            ans_loss = answer_regression_loss(
                answer_pred, answer_true, scale=self.answer_scale
            )

            # 3. Loss de diversidad de cross-attention
            diversity_loss = cross_attention_entropy_loss(attn_weights, inp)

            # Loss total combinado
            total_loss = (
                seq_loss
                + self.answer_weight * ans_loss
                + self.attn_diversity_weight * diversity_loss
            )

        # Gradientes
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        with tf.device('/CPU:0'):
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
            self.train_loss(total_loss)
            self.train_accuracy(accuracy_function(tar_real, predictions))
            self.train_answer_loss(ans_loss)

            # MAE de respuesta (en escala original)
            answer_mae = tf.reduce_mean(
                tf.abs(answer_pred * self.answer_scale - answer_true)
            )
            self.train_answer_mae(answer_mae)

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: Optional[int] = None,
        checkpoint_every: int = 5,
        verbose: bool = True,
        early_stopping_patience: int = 0,
    ) -> dict:
        """
        Loop principal de entrenamiento.

        Espera datasets con ((enc, dec), (target, answer_value)).

        Args:
            train_dataset: Dataset de entrenamiento.
            val_dataset: Dataset de validación (opcional).
            epochs: Número de épocas.
            checkpoint_every: Guardar checkpoint cada N épocas.
            verbose: Imprimir progreso.
            early_stopping_patience: Early stopping por val_loss.

        Returns:
            Historial de métricas.
        """
        if epochs is None:
            epochs = self.config.epochs

        # Restaurar checkpoint
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            if verbose:
                print(f"Checkpoint restaurado: {self.ckpt_manager.latest_checkpoint}")

        if verbose:
            print(f"\nIniciando entrenamiento v3_easy por {epochs} épocas...")
            print(f"  answer_weight (λ): {self.answer_weight}")
            print(f"  attn_diversity_weight (μ): {self.attn_diversity_weight}")
            print(f"  decoder_mask_rate: {self.decoder_mask_rate}")
            print(f"  answer_scale: {self.answer_scale}")
            gpus = tf.config.list_physical_devices('GPU')
            print(f"  Dispositivo: {'GPU' if gpus else 'CPU'}")
            if early_stopping_patience > 0:
                print(f"  Early stopping: patience={early_stopping_patience}")
            print("-" * 60)

        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(epochs):
            start = time.time()

            with tf.device('/CPU:0'):
                self.train_loss.reset_state()
                self.train_accuracy.reset_state()
                self.train_answer_loss.reset_state()
                self.train_answer_mae.reset_state()

            # Iterar sobre batches
            for batch_idx, ((inp, dec_inp), (dec_target, answer_val)) in enumerate(train_dataset):
                self.train_step(inp, dec_inp, dec_target, answer_val)

                if verbose and (batch_idx + 1) % 50 == 0:
                    with tf.device('/CPU:0'):
                        _loss = self.train_loss.result()
                        _acc = self.train_accuracy.result()
                        _ans = self.train_answer_loss.result()
                        _mae = self.train_answer_mae.result()
                    print(f"  Época {epoch+1}, Batch {batch_idx+1}: "
                          f"Loss={_loss:.4f}, Acc={_acc:.4f}, "
                          f"AnsLoss={_ans:.4f}, MAE={_mae:.1f}")

            # Métricas de la época
            with tf.device('/CPU:0'):
                epoch_loss = self.train_loss.result().numpy()
                epoch_acc = self.train_accuracy.result().numpy()
                epoch_ans_loss = self.train_answer_loss.result().numpy()
                epoch_ans_mae = self.train_answer_mae.result().numpy()
            epoch_time = time.time() - start

            self.history["train_loss"].append(float(epoch_loss))
            self.history["train_accuracy"].append(float(epoch_acc))
            self.history["train_answer_loss"].append(float(epoch_ans_loss))
            self.history["train_answer_mae"].append(float(epoch_ans_mae))

            with tf.device('/CPU:0'):
                current_lr = float(self.learning_rate(self.optimizer.iterations))
            self.history["learning_rates"].append(current_lr)

            # Evaluación en validation
            val_loss = val_acc = val_ans_loss = val_ans_mae = None
            if val_dataset is not None:
                val_loss, val_acc, val_ans_loss, val_ans_mae = self.evaluate(val_dataset)
                self.history["val_loss"].append(float(val_loss))
                self.history["val_accuracy"].append(float(val_acc))
                self.history["val_answer_loss"].append(float(val_ans_loss))
                self.history["val_answer_mae"].append(float(val_ans_mae))

            # Logging
            if verbose:
                msg = (f"Época {epoch+1}/{epochs} — "
                       f"Loss: {epoch_loss:.4f} — Acc: {epoch_acc:.4f} — "
                       f"AnsLoss: {epoch_ans_loss:.4f} — MAE: {epoch_ans_mae:.1f}")
                if val_loss is not None:
                    msg += (f" — Val_Loss: {val_loss:.4f} — Val_Acc: {val_acc:.4f} — "
                            f"Val_MAE: {val_ans_mae:.1f}")
                msg += f" — LR: {current_lr:.6f} — {epoch_time:.1f}s"
                print(msg)

            # Checkpoint
            if (epoch + 1) % checkpoint_every == 0:
                ckpt_path = self.ckpt_manager.save()
                if verbose:
                    print(f"  → Checkpoint guardado: {ckpt_path}")

            # Early stopping (basado en val_loss total)
            if early_stopping_patience > 0 and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_path = str(self.checkpoint_dir / "best_model.weights.h5")
                    self.model.save_weights(best_path)
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
                        best_path = str(self.checkpoint_dir / "best_model.weights.h5")
                        if os.path.exists(best_path):
                            self.model.load_weights(best_path)
                        break

        self._save_history()

        if verbose:
            print("-" * 60)
            print("Entrenamiento v3_easy completado!")

        return self.history

    def evaluate(self, dataset: tf.data.Dataset) -> tuple:
        """
        Evalúa en un dataset con answer_value.

        Args:
            dataset: Con estructura ((enc, dec), (target, answer_value)).

        Returns:
            (loss, accuracy, answer_loss, answer_mae).
        """
        losses = []
        accs = []
        ans_losses = []
        ans_maes = []

        @tf.function
        def eval_step(inp, dec_inp, dec_target, answer_true):
            predictions, answer_pred, _ = self.model(
                (inp, dec_inp), training=False, return_attention=True
            )
            loss = loss_function(dec_target, predictions, self.config.label_smoothing)
            acc = accuracy_function(dec_target, predictions)
            ans_loss = answer_regression_loss(
                answer_pred, answer_true, scale=self.answer_scale
            )
            ans_mae = tf.reduce_mean(
                tf.abs(answer_pred * self.answer_scale - answer_true)
            )
            return loss, acc, ans_loss, ans_mae

        for (inp, dec_inp), (dec_target, answer_val) in dataset:
            loss, acc, ans_loss, ans_mae = eval_step(inp, dec_inp, dec_target, answer_val)
            losses.append(float(loss.numpy()))
            accs.append(float(acc.numpy()))
            ans_losses.append(float(ans_loss.numpy()))
            ans_maes.append(float(ans_mae.numpy()))

        import numpy as np
        return (
            np.mean(losses),
            np.mean(accs),
            np.mean(ans_losses),
            np.mean(ans_maes),
        )

    def save_model(self, filepath: Optional[str] = None):
        """Guarda pesos del modelo."""
        if filepath is None:
            filepath = str(self.checkpoint_dir / "model_weights.weights.h5")
        self.model.save_weights(filepath)
        print(f"Pesos guardados en {filepath}")

    def _save_history(self):
        """Guarda historial en JSON."""
        path = self.checkpoint_dir / "training_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
