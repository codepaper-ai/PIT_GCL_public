"""
ESM-2 sequence encoder.

Wraps HuggingFace ESM-2 and returns per-residue embeddings aligned to the
Cα atom ordering from pdb_parser (i.e., no CLS/EOS tokens).
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# ESM-2 hidden sizes per model variant
ESM2_HIDDEN_SIZES = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
}


class ESMEncoder:
    """
    Wraps ESM-2 to produce per-residue embeddings.

    Attributes:
        model_name: HuggingFace model identifier
        hidden_size: Embedding dimensionality of the loaded model
        device: torch device used for inference
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.hidden_size = ESM2_HIDDEN_SIZES.get(model_name, None)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)

        # Resolve hidden size dynamically if not in the lookup table
        if self.hidden_size is None:
            self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, sequence: str, max_length: int = 1022) -> np.ndarray:
        """
        Encode a single amino acid sequence.

        ESM-2 adds CLS and EOS tokens; these are stripped so the output
        length exactly matches len(sequence) (or max_length if truncated).

        Args:
            sequence: Amino acid string (1-letter codes)
            max_length: Maximum sequence length passed to the model.
                        Sequences longer than this are truncated.

        Returns:
            np.ndarray of shape (N, hidden_size), float32
        """
        # Tokenizer will add [CLS] and [EOS]; truncation_length = max_length + 2
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length + 2,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # last_hidden_state: (1, L+2, d)  — strip CLS and EOS
        emb = outputs.last_hidden_state[0, 1:-1].cpu().numpy()  # (L, d)
        return emb.astype(np.float32)

    @torch.no_grad()
    def encode_batch(
        self, sequences: list, max_length: int = 1022
    ) -> list:
        """
        Encode a list of sequences (variable length).

        Returns:
            List of np.ndarray, each shape (N_i, hidden_size)
        """
        return [self.encode(seq, max_length=max_length) for seq in sequences]
