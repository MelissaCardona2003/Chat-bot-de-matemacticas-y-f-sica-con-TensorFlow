"""
tokenizer.py — Tokenización a nivel de carácter.

Implementa un tokenizador que opera a nivel de carácter individual,
con tokens especiales <PAD>, <START>, <END>, <UNK>.

Uso:
    from transformer_math_physics_tutor.data.tokenizer import CharTokenizer

    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["hello", "world"])
    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


class CharTokenizer:
    """
    Tokenizador a nivel de carácter para secuencias matemáticas.

    Cada carácter individual se mapea a un índice numérico.
    Incluye tokens especiales obligatorios:
        - <PAD> = 0 (padding)
        - <START> = 1 (inicio de secuencia)
        - <END> = 2 (fin de secuencia)
        - <UNK> = 3 (carácter desconocido)

    Attributes:
        char_to_idx: Diccionario de carácter a índice.
        idx_to_char: Diccionario de índice a carácter.
        special_tokens: Diccionario de tokens especiales.
    """

    # Tokens especiales con índices fijos
    PAD_TOKEN = "<PAD>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    UNK_TOKEN = "<UNK>"

    PAD_ID = 0
    START_ID = 1
    END_ID = 2
    UNK_ID = 3

    def __init__(self, vocab_file: Optional[str] = None):
        """
        Inicializa el tokenizador.

        Args:
            vocab_file: Ruta a un archivo de vocabulario JSON previamente guardado.
                        Si se proporciona, carga el vocabulario desde el archivo.
        """
        self.special_tokens = {
            self.PAD_TOKEN: self.PAD_ID,
            self.START_TOKEN: self.START_ID,
            self.END_TOKEN: self.END_ID,
            self.UNK_TOKEN: self.UNK_ID,
        }

        self.char_to_idx: Dict[str, int] = dict(self.special_tokens)
        self.idx_to_char: Dict[int, str] = {v: k for k, v in self.char_to_idx.items()}

        if vocab_file is not None:
            self.load_vocab(vocab_file)

    @property
    def vocab_size(self) -> int:
        """
        Tamaño total del vocabulario (incluyendo tokens especiales).

        Returns:
            Número de tokens en el vocabulario.
        """
        return len(self.char_to_idx)

    @property
    def pad_token_id(self) -> int:
        """ID del token de padding."""
        return self.PAD_ID

    @property
    def start_token_id(self) -> int:
        """ID del token de inicio."""
        return self.START_ID

    @property
    def end_token_id(self) -> int:
        """ID del token de fin."""
        return self.END_ID

    @property
    def unk_token_id(self) -> int:
        """ID del token desconocido."""
        return self.UNK_ID

    def build_vocab(self, texts: List[str]) -> None:
        """
        Construye el vocabulario a partir de una lista de textos.

        Extrae todos los caracteres únicos de los textos proporcionados
        y les asigna índices, comenzando después de los tokens especiales.

        Args:
            texts: Lista de strings a partir de los cuales construir el vocabulario.

        Returns:
            None
        """
        # Recolectar todos los caracteres únicos
        all_chars = set()
        for text in texts:
            all_chars.update(set(text))

        # Ordenar para reproducibilidad
        sorted_chars = sorted(all_chars)

        # Reiniciar vocabulario con tokens especiales
        self.char_to_idx = dict(self.special_tokens)
        next_idx = len(self.special_tokens)  # Empezar después de tokens especiales

        for char in sorted_chars:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = next_idx
                next_idx += 1

        # Reconstruir mapa inverso
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        print(f"Vocabulario construido: {self.vocab_size} tokens "
              f"({self.vocab_size - len(self.special_tokens)} caracteres + "
              f"{len(self.special_tokens)} especiales)")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Codifica un texto en una secuencia de índices.

        Args:
            text: Texto a codificar.
            add_special_tokens: Si True, añade <START> al inicio y <END> al final.

        Returns:
            Lista de índices enteros.
        """
        indices = []

        if add_special_tokens:
            indices.append(self.START_ID)

        for char in text:
            idx = self.char_to_idx.get(char, self.UNK_ID)
            indices.append(idx)

        if add_special_tokens:
            indices.append(self.END_ID)

        return indices

    def decode(
        self,
        indices: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decodifica una secuencia de índices en texto.

        Args:
            indices: Lista de índices enteros.
            skip_special_tokens: Si True, omite tokens especiales (<PAD>, <START>, <END>).

        Returns:
            Texto decodificado.
        """
        chars = []
        special_ids = {self.PAD_ID, self.START_ID, self.END_ID}

        for idx in indices:
            # Convertir a int si es tensor
            if hasattr(idx, "numpy"):
                idx = int(idx.numpy())
            else:
                idx = int(idx)

            if skip_special_tokens and idx in special_ids:
                continue

            char = self.idx_to_char.get(idx, "")
            chars.append(char)

        return "".join(chars)

    def save_vocab(self, filepath: str) -> None:
        """
        Guarda el vocabulario en un archivo JSON.

        Args:
            filepath: Ruta del archivo JSON de salida.

        Returns:
            None
        """
        vocab_data = {
            "char_to_idx": self.char_to_idx,
            "special_tokens": self.special_tokens,
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        print(f"Vocabulario guardado en {filepath} ({self.vocab_size} tokens)")

    def load_vocab(self, filepath: str) -> None:
        """
        Carga el vocabulario desde un archivo JSON.

        Args:
            filepath: Ruta del archivo JSON del vocabulario.

        Returns:
            None
        """
        with open(filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.char_to_idx = vocab_data["char_to_idx"]
        # Convertir claves de idx_to_char a int (JSON las guarda como string)
        self.idx_to_char = {int(v): k for k, v in self.char_to_idx.items()}

        if "special_tokens" in vocab_data:
            self.special_tokens = vocab_data["special_tokens"]

        print(f"Vocabulario cargado desde {filepath} ({self.vocab_size} tokens)")

    def __repr__(self) -> str:
        return (
            f"CharTokenizer(vocab_size={self.vocab_size}, "
            f"special_tokens={list(self.special_tokens.keys())})"
        )


if __name__ == "__main__":
    # Ejemplo de uso
    print("=" * 60)
    print("DEMO: CharTokenizer")
    print("=" * 60)

    tokenizer = CharTokenizer()

    # Construir vocabulario con textos de ejemplo
    sample_texts = [
        "Solve for x: 2x + 3 = 7",
        "Find the value of (3/4) + (1/2)",
        "What is sqrt(16)?",
        "x^2 - 5x + 6 = 0",
    ]
    tokenizer.build_vocab(sample_texts)

    # Codificar y decodificar
    test_text = "2x + 3 = 7"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\nTexto original: '{test_text}'")
    print(f"Codificado: {encoded}")
    print(f"Decodificado: '{decoded}'")
    print(f"Coincide: {test_text == decoded}")
    print(f"\n{tokenizer}")
