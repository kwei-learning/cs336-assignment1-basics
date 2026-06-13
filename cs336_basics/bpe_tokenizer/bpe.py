from __future__ import annotations

import heapq
import os
import regex as re
from collections import Counter
from multiprocessing import Pool

from cs336_basics.pretokenization_example import find_chunk_boundaries

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
    str_counts: Counter[str] = Counter()
    for chunk in _split_on_special_tokens(text, special_tokens):
        if chunk == "" or chunk in special_set:
            continue
        str_counts.update(match.group() for match in _COMPILED_PAT.finditer(chunk))

    counts: Counter[tuple[bytes, ...]] = Counter()
    for pretoken, c in str_counts.items():
        token_bytes = pretoken.encode("utf-8")
        counts[tuple(bytes([b]) for b in token_bytes)] += c
    return counts


def _count_chunk(args: tuple[str, int, int, list[str]]) -> Counter[tuple[bytes, ...]]:
    """Worker: pre-token counts for one [start, end) byte range of a file.

    Top-level (picklable) so it can run in a multiprocessing pool. Boundaries are
    aligned to a special token by ``find_chunk_boundaries`` so no pre-token is
    split across a chunk edge; summing the per-chunk counters reproduces the
    serial result exactly.
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return _pretoken_counts(chunk, special_tokens)


def _parallel_pretoken_counts(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_procs: int | None = None,
) -> Counter[tuple[bytes, ...]]:
    """Pre-token counts over the whole file, parallelized across processes.

    Falls back to a serial single-read path when parallelism would not help
    (no special token to split on, a single core, or a tiny/non-chunkable file).
    """
    if num_procs is None:
        num_procs = os.cpu_count() or 1

    # We need a special token to find safe chunk boundaries; without one we can't
    # guarantee a chunk edge doesn't fall inside a pre-token, so read serially.
    split_token = special_tokens[0].encode("utf-8") if special_tokens else None
    if num_procs <= 1 or split_token is None:
        with open(input_path, encoding="utf-8") as f:
            return _pretoken_counts(f.read(), special_tokens)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_procs, split_token)

    # find_chunk_boundaries may collapse to fewer (or one) boundary pair.
    chunk_args = [
        (str(input_path), start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    if len(chunk_args) <= 1:
        return _count_chunk(chunk_args[0]) if chunk_args else Counter()

    total: Counter[tuple[bytes, ...]] = Counter()
    with Pool(min(num_procs, len(chunk_args))) as pool:
        for partial in pool.map(_count_chunk, chunk_args):
            total.update(partial)
    return total


class _HeapEntry:
    """Heap entry ordered so the BPE-best pair pops first from a min-heap.

    "Best" means highest count, ties broken by the lexicographically greatest
    pair -- identical to ``max(pair_counts, key=lambda p: (pair_counts[p], p))``.
    """

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: tuple[bytes, bytes]) -> None:
        self.count = count
        self.pair = pair

    def __lt__(self, other: "_HeapEntry") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


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

    # Pre-tokenization is the dominant cost on large corpora, so parallelize it
    # across processes (boundaries aligned to a special token keep results exact).
    pretoken_counts = _parallel_pretoken_counts(input_path, special_tokens)
    # print(pretoken_counts)
    # Sample output:
    # (b' ', b'h', b'o', b'm', b'e', b'w', b'o', b'r', b'k'): 5, the word homework appears 5 times as pre-token

    # Represent each unique pre-token as a list of byte tokens with its frequency.
    sequences: list[list[bytes]] = [list(seq) for seq in pretoken_counts]
    freqs: list[int] = list(pretoken_counts.values())

    merges: list[tuple[bytes, bytes]] = []
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_seqs: dict[tuple[bytes, bytes], set[int]] = {}
    heap: list[_HeapEntry] = []

    def add_seq_pairs(idx: int) -> None:
        seq = sequences[idx]
        freq = freqs[idx]
        for pair in zip(seq, seq[1:]):
            pair_counts[pair] += freq
            pair_to_seqs.setdefault(pair, set()).add(idx)
            heapq.heappush(heap, _HeapEntry(pair_counts[pair], pair))

    def remove_seq_pairs(idx: int) -> None:
        seq = sequences[idx]
        freq = freqs[idx]
        for pair in zip(seq, seq[1:]):
            pair_counts[pair] -= freq
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
            else:
                heapq.heappush(heap, _HeapEntry(pair_counts[pair], pair))
            seqs = pair_to_seqs.get(pair)
            if seqs is not None:
                seqs.discard(idx)
                if not seqs:
                    del pair_to_seqs[pair]

    for idx in range(len(sequences)):
        add_seq_pairs(idx)

    while len(vocab) < vocab_size:
        # Pop the best pair, discarding stale entries (count no longer current).
        best_pair = None
        while heap:
            entry = heapq.heappop(heap)
            if pair_counts.get(entry.pair) == entry.count:
                best_pair = entry.pair
                break
        if best_pair is None:
            break
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
