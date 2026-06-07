from __future__ import annotations

import os
import regex as re
from collections import Counter

# GPT-2 pre-tokenization pattern.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(PAT)


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split text on special tokens, keeping the special tokens as standalone chunks."""
    if not special_tokens:
        return [text]
    # Sort by length descending so longer (possibly overlapping) tokens match first.
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(tok) for tok in sorted_specials)
    return re.split(f"({pattern})", text)


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """Count pre-token frequencies, where each pre-token is a tuple of single-byte bytes.

    Special tokens are stripped out entirely (they never participate in merges).
    """
    special_set = set(special_tokens)
    counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in _split_on_special_tokens(text, special_tokens):
        if chunk == "" or chunk in special_set:
            continue
        for match in _COMPILED_PAT.finditer(chunk):
            token_bytes = match.group().encode("utf-8")
            counts[tuple(bytes([b]) for b in token_bytes)] += 1
    return counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns the vocabulary (id -> bytes) and the ordered list of merges.
    """
    # Initialize vocab with all 256 single bytes, then special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # print(text)
    # print(special_tokens) -> ['<|endoftext|>']

    pretoken_counts = _pretoken_counts(text, special_tokens)
    # print(pretoken_counts)
    # Sample output:
    # (b' ', b'h', b'o', b'm', b'e', b'w', b'o', b'r', b'k'): 5, the word homework appears 5 times as pre-token

    # Represent each unique pre-token as a list of byte tokens with its frequency.
    sequences: list[list[bytes]] = [list(seq) for seq in pretoken_counts]
    freqs: list[int] = list(pretoken_counts.values())

    merges: list[tuple[bytes, bytes]] = []

    # Incremental bookkeeping, built once and then updated in place:
    #   pair_counts : adjacent byte-pair -> total (frequency-weighted) count
    #   pair_to_seqs: adjacent byte-pair -> set of sequence indices containing it
    # Each merge only touches the sequences that actually contain the chosen pair,
    # so we avoid rescanning / rewriting the whole corpus every iteration.
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_seqs: dict[tuple[bytes, bytes], set[int]] = {}

    def add_seq_pairs(idx: int) -> None:
        seq = sequences[idx]
        freq = freqs[idx]
        for pair in zip(seq, seq[1:]):
            pair_counts[pair] += freq
            pair_to_seqs.setdefault(pair, set()).add(idx)

    def remove_seq_pairs(idx: int) -> None:
        seq = sequences[idx]
        freq = freqs[idx]
        for pair in zip(seq, seq[1:]):
            pair_counts[pair] -= freq
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
            seqs = pair_to_seqs.get(pair)
            if seqs is not None:
                seqs.discard(idx)
                if not seqs:
                    del pair_to_seqs[pair]

    for idx in range(len(sequences)):
        add_seq_pairs(idx)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        # Most frequent pair; ties broken by lexicographically greatest pair.
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merged = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged
        merges.append(best_pair)

        # Only the sequences containing best_pair can change. For each, drop its
        # old pair contributions, rewrite it, then add the new pair contributions.
        for idx in list(pair_to_seqs.get(best_pair, ())):
            remove_seq_pairs(idx)
            seq = sequences[idx]
            new_seq: list[bytes] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            sequences[idx] = new_seq
            add_seq_pairs(idx)

    return vocab, merges
