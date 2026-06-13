from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.bpe_tokenizer.bpe import (
    _COMPILED_PAT,
    _split_on_special_tokens,
)


class Tokenizer:
    """Byte-level BPE tokenizer that encodes text to token IDs and back.

    The tokenizer is parameterized by a vocabulary (id -> bytes), an ordered
    list of merges, and an optional list of special-token strings that are
    never split during encoding.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # Make sure every special token has an id. Some vocabs already include
        # them; otherwise append at the end.
        existing = set(self.vocab.values())
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in existing:
                self.vocab[len(self.vocab)] = tok_bytes
                existing.add(tok_bytes)

        # Inverse vocab for O(1) bytes -> id lookups during encode.
        self.token_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # Merge priority: lower rank == merged earlier == higher priority.
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Construct a Tokenizer from a serialized vocab (JSON) and merges (txt).

        The vocab JSON maps token strings to ids; the merges file has one
        space-separated pair per line. Both use the bytes' direct utf-8 decoding
        as written by our own training output.
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(idx): token.encode("utf-8") for token, idx in raw_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip("\n")
                parts = cleaned.split(" ")
                if len(parts) == 2 and parts[0] and parts[1]:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def _merge_pretoken(self, token_bytes: bytes) -> list[int]:
        """Apply BPE merges to a single pre-token and return its token ids."""
        # Start from individual bytes.
        parts: list[bytes] = [bytes([b]) for b in token_bytes]

        while len(parts) > 1:
            # Find the adjacent pair with the lowest merge rank.
            best_rank = None
            best_idx = -1
            for i in range(len(parts) - 1):
                rank = self.merge_ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            parts[best_idx : best_idx + 2] = [parts[best_idx] + parts[best_idx + 1]]

        return [self.token_to_id[p] for p in parts]

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token ids."""
        special_set = set(self.special_tokens)
        ids: list[int] = []

        # Split keeps special tokens as standalone chunks; longest-match-first
        # is handled inside _split_on_special_tokens.
        for chunk in _split_on_special_tokens(text, self.special_tokens):
            if chunk == "":
                continue
            if chunk in special_set:
                ids.append(self.token_to_id[chunk.encode("utf-8")])
                continue
            for match in _COMPILED_PAT.finditer(chunk):
                ids.extend(self._merge_pretoken(match.group().encode("utf-8")))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g. a file handle).

        Yields token ids one at a time so we never hold the whole corpus in
        memory.
        """
        for piece in iterable:
            yield from self.encode(piece)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a string."""
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8", errors="replace")
