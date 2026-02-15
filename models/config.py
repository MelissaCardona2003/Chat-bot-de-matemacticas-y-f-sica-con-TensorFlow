"""
config.py — Hiperparámetros del Transformer.

Centraliza todos los hiperparámetros del modelo en una clase
de configuración para fácil ajuste y serialización.

Uso:
    from transformer_math_physics_tutor.models.config import TransformerConfig
    config = TransformerConfig(d_model=256, num_heads=8)
"""

import json
from pathlib import Path


class TransformerConfig:
    """
    Configuración de hiperparámetros del Transformer.

    Contiene todos los parámetros necesarios para instanciar
    y entrenar el modelo Transformer Encoder-Decoder.

    Attributes:
        d_model: Dimensión de los embeddings y capas internas.
        num_heads: Número de cabezas de atención.
        num_layers: Número de capas de encoder y decoder.
        dff: Dimensión de la capa feed-forward interna.
        dropout_rate: Tasa de dropout.
        max_encoder_len: Longitud máxima de secuencia del encoder.
        max_decoder_len: Longitud máxima de secuencia del decoder.
        vocab_size: Tamaño del vocabulario (se define tras construir tokenizer).
        warmup_steps: Pasos de warmup para el scheduler.
        label_smoothing: Factor de suavizado de etiquetas.
        epochs: Número de épocas de entrenamiento.
        batch_size: Tamaño del batch.
        checkpoint_dir: Directorio para guardar checkpoints.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dff: int = 1024,
        dropout_rate: float = 0.1,
        max_encoder_len: int = 100,
        max_decoder_len: int = 150,
        vocab_size: int = 128,
        warmup_steps: int = 4000,
        lr_scale: float = 1.0,
        label_smoothing: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        checkpoint_dir: str = "checkpoints"
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.vocab_size = vocab_size
        self.warmup_steps = warmup_steps
        self.lr_scale = lr_scale
        self.label_smoothing = label_smoothing
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir

        # Verificación de consistencia
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) debe ser divisible por num_heads ({num_heads})"
        )

    @property
    def depth(self) -> int:
        """Dimensión por cabeza de atención: d_model / num_heads."""
        return self.d_model // self.num_heads

    def to_dict(self) -> dict:
        """
        Convierte la configuración a diccionario.

        Returns:
            Diccionario con todos los hiperparámetros.
        """
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
            "max_encoder_len": self.max_encoder_len,
            "max_decoder_len": self.max_decoder_len,
            "vocab_size": self.vocab_size,
            "warmup_steps": self.warmup_steps,
            "label_smoothing": self.label_smoothing,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "checkpoint_dir": self.checkpoint_dir,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TransformerConfig":
        """
        Crea una instancia desde un diccionario.

        Args:
            config_dict: Diccionario con hiperparámetros.

        Returns:
            Instancia de TransformerConfig.
        """
        return cls(**config_dict)

    def save(self, filepath: str) -> None:
        """
        Guarda la configuración en archivo JSON.

        Args:
            filepath: Ruta del archivo JSON de salida.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuración guardada en {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TransformerConfig":
        """
        Carga la configuración desde archivo JSON.

        Args:
            filepath: Ruta del archivo JSON.

        Returns:
            Instancia de TransformerConfig.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        print(f"Configuración cargada desde {filepath}")
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"TransformerConfig({params})"
