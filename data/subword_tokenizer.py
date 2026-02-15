"""
subword_tokenizer.py — Tokenización por subpalabras (BPE via SentencePiece).

Versión 2 del tokenizer: reemplaza el CharTokenizer (nivel carácter)
por un tokenizer de subpalabras que produce secuencias ~4–5× más cortas.

¿Por qué subpalabras en vez de caracteres?
- Con char-level, "Step 1: 5 + 3 = 8" → 18 tokens.
  Con BPE vocab=4000, → ~10 tokens.
- Secuencias más cortas significan que la cross-attention del
  decoder puede "ver" todo el encoder con menos pasos, haciendo
  más difícil el atajo autoregresivo que colapsaba la v1.
- El modelo opera a nivel de conceptos (palabras/subpalabras)
  en vez de caracteres individuales.

Tokens especiales (mismos IDs que CharTokenizer para compatibilidad):
    <PAD>   = 0
    <START> = 1   (BOS en SentencePiece)
    <END>   = 2   (EOS en SentencePiece)
    <UNK>   = 3

Uso:
    from transformer_math_physics_tutor.data.subword_tokenizer import SubwordTokenizer

    tokenizer = SubwordTokenizer()
    tokenizer.train(texts, vocab_size=4000)
    tokenizer.save("checkpoints/v2_subword/sp")
    # O cargar uno existente:
    tokenizer = SubwordTokenizer("checkpoints/v2_subword/sp.model")
    encoded = tokenizer.encode("What is 5 + 3?")
    decoded = tokenizer.decode(encoded)
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Optional


class SubwordTokenizer:
    """
    Tokenizador de subpalabras basado en SentencePiece BPE.

    Usa el mismo contrato de API que CharTokenizer:
        - encode(text, add_special_tokens=True) → List[int]
        - decode(ids, skip_special_tokens=True) → str
        - save_vocab(path) / load_vocab(path)
        - Propiedades: vocab_size, pad_token_id, start_token_id, etc.

    Internamente usa SentencePiece con:
        pad_id=0 (<PAD>), bos_id=1 (<START>), eos_id=2 (<END>), unk_id=3 (<UNK>)

    Attributes:
        sp: Instancia de SentencePieceProcessor.
        _vocab_size: Tamaño del vocabulario (cache).
    """

    # Tokens especiales — mismos IDs que CharTokenizer
    PAD_TOKEN = "<PAD>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    UNK_TOKEN = "<UNK>"

    PAD_ID = 0
    START_ID = 1
    END_ID = 2
    UNK_ID = 3

    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el tokenizador.

        Args:
            model_path: Ruta al archivo .model de SentencePiece.
                        Si se proporciona, carga el modelo entrenado.
                        Si None, se debe llamar a train() antes de usar.
        """
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self._vocab_size = 0
        self._model_path = model_path

        if model_path is not None:
            self.load_vocab(model_path)

    @property
    def vocab_size(self) -> int:
        """Tamaño total del vocabulario (incluyendo tokens especiales)."""
        return self._vocab_size

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

    def train(
        self,
        texts: List[str],
        vocab_size: int = 4000,
        model_prefix: str = "/tmp/sp_tokenizer",
        character_coverage: float = 1.0,
    ) -> None:
        """
        Entrena el tokenizador BPE sobre una lista de textos.

        Args:
            texts: Lista de strings (problemas + soluciones del dataset).
            vocab_size: Tamaño del vocabulario BPE (recomendado: 2000–4000).
            model_prefix: Prefijo para los archivos de salida (.model y .vocab).
            character_coverage: Cobertura de caracteres (1.0 = todos).
        """
        import sentencepiece as spm

        # Escribir textos a archivo temporal (SentencePiece requiere archivo)
        tmp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        )
        try:
            for text in texts:
                line = text.strip().replace('\n', ' ')
                if line:
                    tmp_file.write(line + '\n')
            tmp_file.close()

            # Entrenar SentencePiece
            spm.SentencePieceTrainer.train(
                input=tmp_file.name,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type='bpe',
                character_coverage=character_coverage,
                # Tokens especiales con IDs fijos (compatibles con CharTokenizer)
                pad_id=self.PAD_ID,
                bos_id=self.START_ID,
                eos_id=self.END_ID,
                unk_id=self.UNK_ID,
                pad_piece=self.PAD_TOKEN,
                bos_piece=self.START_TOKEN,
                eos_piece=self.END_TOKEN,
                unk_piece=self.UNK_TOKEN,
                # Config adicional
                max_sentence_length=4096,
                num_threads=4,
                train_extremely_large_corpus=False,
                # Tratar dígitos como tokens individuales para
                # que el modelo aprenda aritmética
                byte_fallback=True,
                split_digits=True,
            )

            # Cargar modelo entrenado
            self.sp.load(model_prefix + '.model')
            self._vocab_size = self.sp.get_piece_size()
            self._model_path = model_prefix + '.model'

            print(f"SubwordTokenizer entrenado: {self._vocab_size} tokens")
            print(f"  Modelo guardado en: {model_prefix}.model")

        finally:
            os.unlink(tmp_file.name)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Codifica un texto en una secuencia de IDs de subpalabra.

        Args:
            text: Texto a codificar.
            add_special_tokens: Si True, añade <START> al inicio y <END> al final.

        Returns:
            Lista de índices enteros.
        """
        # SentencePiece encode (sin tokens especiales — los añadimos manualmente)
        ids = self.sp.encode(text, out_type=int)

        if add_special_tokens:
            ids = [self.START_ID] + ids + [self.END_ID]

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decodifica una secuencia de IDs en texto.

        Args:
            ids: Lista de IDs de subpalabra.
            skip_special_tokens: Si True, omite <PAD>, <START>, <END>.

        Returns:
            Texto decodificado.
        """
        # Convertir a ints nativos
        clean_ids = []
        special_ids = {self.PAD_ID, self.START_ID, self.END_ID}

        for idx in ids:
            if hasattr(idx, 'numpy'):
                idx = int(idx.numpy())
            else:
                idx = int(idx)

            if skip_special_tokens and idx in special_ids:
                continue
            clean_ids.append(idx)

        return self.sp.decode(clean_ids)

    def save_vocab(self, filepath: str) -> None:
        """
        Guarda el modelo del tokenizador.

        Copia el archivo .model de SentencePiece al destino especificado.
        También guarda un JSON con metadatos (vocab_size, special tokens).

        Args:
            filepath: Ruta base (sin extensión) para los archivos de salida.
                      Se crearán filepath.model y filepath.json.
        """
        import shutil

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Copiar .model
        model_dest = str(path) if str(path).endswith('.model') else str(path) + '.model'
        if self._model_path and os.path.exists(self._model_path):
            src = os.path.abspath(self._model_path)
            dst = os.path.abspath(model_dest)
            if src != dst:
                shutil.copy2(src, dst)

        # Copiar .vocab si existe
        vocab_src = self._model_path.replace('.model', '.vocab') if self._model_path else None
        if vocab_src and os.path.exists(vocab_src):
            vocab_dest = model_dest.replace('.model', '.vocab')
            src_v = os.path.abspath(vocab_src)
            dst_v = os.path.abspath(vocab_dest)
            if src_v != dst_v:
                shutil.copy2(src_v, dst_v)

        # Guardar metadatos JSON
        meta_path = model_dest.replace('.model', '_meta.json')
        meta = {
            "tokenizer_type": "subword_bpe",
            "vocab_size": self._vocab_size,
            "special_tokens": {
                self.PAD_TOKEN: self.PAD_ID,
                self.START_TOKEN: self.START_ID,
                self.END_TOKEN: self.END_ID,
                self.UNK_TOKEN: self.UNK_ID,
            },
            "model_file": os.path.basename(model_dest),
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        print(f"SubwordTokenizer guardado en {model_dest} ({self._vocab_size} tokens)")

    def load_vocab(self, filepath: str) -> None:
        """
        Carga el modelo del tokenizador desde un archivo .model.

        Args:
            filepath: Ruta al archivo .model de SentencePiece.
        """
        model_path = filepath if filepath.endswith('.model') else filepath + '.model'
        self.sp.load(model_path)
        self._vocab_size = self.sp.get_piece_size()
        self._model_path = model_path
        print(f"SubwordTokenizer cargado desde {model_path} ({self._vocab_size} tokens)")

    def id_to_piece(self, token_id: int) -> str:
        """Convierte un ID a su representación de subpalabra."""
        return self.sp.id_to_piece(token_id)

    def piece_to_id(self, piece: str) -> int:
        """Convierte una subpalabra a su ID."""
        return self.sp.piece_to_id(piece)

    def build_vocab(self, texts: List[str], vocab_size: int = 4000) -> None:
        """
        Alias de train() para compatibilidad con CharTokenizer.

        Args:
            texts: Lista de strings para entrenar el vocabulario.
            vocab_size: Tamaño del vocabulario.
        """
        self.train(texts, vocab_size=vocab_size)

    def __repr__(self) -> str:
        return (
            f"SubwordTokenizer(vocab_size={self._vocab_size}, "
            f"type=BPE, special_tokens=[PAD, START, END, UNK])"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: SubwordTokenizer")
    print("=" * 60)

    tokenizer = SubwordTokenizer()

    sample_texts = [
        "Janet has 3 apples and buys 5 more. How many apples does she have?",
        "Step 1: Add 3 + 5 = 8.\nAnswer: 8",
        "A car accelerates from rest at 3 m/s² for 5 seconds.",
        "Use the formula v = v₀ + at.\nStep 1: v = 0 + 3 × 5 = 15.\nAnswer: 15 m/s",
        "What is the kinetic energy of a 4 kg object moving at 5 m/s?",
        "Use KE = ½mv².\nStep 1: KE = 0.5 × 4 × 5² = 50.\nAnswer: 50 J",
    ]

    tokenizer.train(sample_texts, vocab_size=200)

    for text in sample_texts[:3]:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"\nOriginal:  '{text}'")
        print(f"Encoded:   {encoded} ({len(encoded)} tokens)")
        print(f"Decoded:   '{decoded}'")
        print(f"Match:     {text.strip() == decoded.strip()}")

    print(f"\n{tokenizer}")
