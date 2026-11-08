import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Iterable, Tuple

class ByteLevelBPETokenizer:
    """

    Production Byte-Level BPE Tokenizer built from scratch.
    Guarantees zero out-of-vocabulary (<unk>) tokens by backing off to raw UTF-8 bytes.
    """

    PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    SPECIAL_TOKENS = {
        "<|pad|>": 256,
        "<|bos|>": 257,
        "<|eos|>": 258,
        "<|user|>": 259,
        "<|assistant|>": 260,
        "<|system|>": 261,
    }

    def __init__(self) -> None:
        self.merges: Dict[Tuple[str, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.inverse_special_tokens: Dict[int, str] = {
            v: k for k, v in self.special_tokens.items()
        }

        self.chunk_pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+"
        )
    @property
    def pad_id(self) -> int:
        return self.special_tokens["<|pad|>"]
    @property
    def bos_id(self) -> int:
        return self.special_tokens["<|bos|>"]
    @property
    def eos_id(self) -> int:
        return self.special_tokens["<|eos|>"]

    @staticmethod
    def _get_pair_stats(word_splits: List[List[int]]) -> Counter:
        counts: Counter = Counter()
        for ids in word_splits:
            for pair in zip(ids, ids[1:]):
                counts[pair] += 1
        return counts

    @staticmethod
    def _merge_pair(word_splits: List[List[int]], pair: Tuple[int, int], new_id: int) -> List[List[int]]:
        p0, p1 = pair
        updated_splits: List[List[int]] = []
        for ids in word_splits:
            new_ids: List[int] = []
            i = 0
            n = len(ids)
            while i < n:
                if i < n - 1 and ids[i] == p0 and ids[i+1] == p1:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            updated_splits.append(new_ids)
        return updated_splits

    def train(self, corpus: Iterable[str], target_vocab_size: int = 2048) -> None:
        """Trains BPE merge rules from an iterable text corpus."""
        base_offset = 256 + len(self.special_tokens)
        num_merges = max(0, target_vocab_size - base_offset)

        word_splits: List[List[int]] = []
        for text in corpus:
            for chunk in self._chunk_pattern.findall(text):
                word_splits.append(list(chunk.encode("utf-8")))
        for idx in range(num_merges):
            stats = self._get_pair_stats(word_splits)
            if not stats:
                break
            best_pair, freq = stats.most_common(1)[0]
            if freq < 2:
                break
            new_id = base_offset + idx
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            word_splits = self._merge_pair(word_splits, best_pair, new_id)

    def _encode_chunk(self, raw_bytes: bytes) -> List[int]:
        ids = list(raw_bytes)
        while len(ids) >= 2:
            candidate_pairs = [(ids[i], ids[i+1]) for i in range(len(ids) - 1)]
            valid_merges = {
                pair: self.merges[pair]
                for pair in candidate_pairs
                if pair in self.merges
            }
            if not valid_merges:
                break
            best_pair = min(valid_merges, key=lambda p: valid_merges[p])
            new_id = valid_merges[best_pair]
            ids = self._merge_pair([ids], best_pair, new_id)[0]
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encodes string while respecting atomic special tokens."""
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        segments = re.split(special_pattern, text)

        token_ids: List[int] = []
        if add_bos:
            token_ids.append(self.bos_id)

        for segment in segments:
            if not segment:
                continue
            if segment in self.special_tokens:
                token_ids.append(self.special_tokens[segment])
            else:
                for sub_chunk in self._chunk_pattern.findall(segment):
                    token_ids.extend(self._encode_chunk(sub_chunk.encode("utf-8")))

        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token Ids back into a valid UTF-8 string."""
        output_parts: List[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                output_parts.append(byte_buffer.decode("utf-8", error="replace"))
                byte_buffer.clear()

        for tid in token_ids:
            if tid in self.inverse_special_tokens:
                flush_bytes()
                if not skip_special_tokens:
                    output_parts.append(self.inverse_special_tokens[tid])
            elif tid in self.vocab:
                byte_buffer.extend(self.vocab[tid])
            elif 0 <= tid < 256:
                byte_buffer.append(tid)
        flush_bytes()
        return "".join(output_parts)

    def save(self, path: Path) -> None:
        serialized = {
            "merges": [[p0, p1, nid] for (p0, p1), nid in self.merges.items()],
            "special_tokens": self.special_tokens
        }
        path.write_text(json.dumps(serialized, indent=2), encoding='utf-8')

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.special_tokens = data["special_tokens"]
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.merges.clear()
        self.vocab = {i: bytes([i]) for i in range(256)}
        for p0, p1, nid in data["merges"]:
            self.merges[(p0, p1)] = nid
            self.vocab[nid] = self.vocab[p0] + self.vocab[p1]