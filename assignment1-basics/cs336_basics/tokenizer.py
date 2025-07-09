# cs336_basics/tokenizer.py

from typing import Iterable, Iterator, List, Dict, Tuple
import os
import regex as re
from array import array
import heapq
from collections import defaultdict, Counter
from functools import total_ordering
import json

GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> list[bytes]:
    """使用GPT-2的正则表达式将文本分割成“词块”，并编码为bytes。 This step is very important!!!! Otherwise the b'a\n\nb' will be transfer into 'a' '\n\n' 'b' instead of 'a' '\n' '\n' 'b'"""
    str_tokens = re.findall(GPT2_SPLIT_PATTERN, text)
    byte_tokens = [s.encode("utf-8") for s in str_tokens]
    return byte_tokens


GPT2_RE = re.compile(GPT2_SPLIT_PATTERN)


def iter_pretokenize(text: str) -> Iterator[bytes]:
    """按 GPT-2 正则逐个产生字节串，零内存列表。"""
    for m in GPT2_RE.finditer(text):
        yield m.group(0).encode("utf-8")


class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str] | None = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [
            token.encode("utf-8") for token in self.special_tokens
        ]

        self.merges: List[Tuple[bytes, bytes]] = []
        self.stoi: Dict[bytes, int] = {}
        self.itos: Dict[int, bytes] = {}
        self.merges_rank: Dict[Tuple[bytes, bytes], int] = {}

        # init vocab
        for i, token_bytes in enumerate(self.special_tokens_bytes):  # special tokens
            self.stoi[token_bytes] = i
            self.itos[i] = token_bytes

        offset = len(self.special_tokens_bytes)  # 单字节 tokens
        for i in range(256):
            self.stoi[bytes([i])] = i + offset
            self.itos[i + offset] = bytes([i])

        self.vocab = self.itos.copy()  # for serialization
        self.merges_rank = {}  # for fast lookup
        # pair2new: (p1, p2) -> new_token_id
        self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in self.merges}

    def _get_stats(self, token_groups: list[list[int]]):
        """Count the frequency of occurrence of all byte pairs."""
        pair_counts = {}
        for group in token_groups:
            for pair in zip(group, group[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    def _merge_pair_in_groups(
        self, ids_group: list[list[int]], pair_to_merge: tuple[int, int], new_id: int
    ):
        """One merge in vocab"""
        new_ids_group = []
        for group in ids_group:
            new_group = []
            i = 0
            while i < len(group):
                if i < len(group) - 1 and (group[i], group[i + 1]) == pair_to_merge:
                    new_group.append(new_id)
                    i += 2
                else:
                    new_group.append(group[i])
                    i += 1
            new_ids_group.append(new_group)
        return new_ids_group


    def train(self, path: str | os.PathLike):
        """使用自定义类实现大根堆的 BPE 训练"""
        
        class PairItem:
            """自定义类用于在堆中实现正确的排序"""
            def __init__(self, count, token_id1, token_id2, itos):
                self.count = count
                self.token_id1 = token_id1
                self.token_id2 = token_id2
                self.itos = itos
                self.bytes1 = itos[token_id1]
                self.bytes2 = itos[token_id2]
            
            def __lt__(self, other):
                # 首先按频次降序（大的在前）
                if self.count != other.count:
                    return self.count > other.count
                # 频次相同时，按第一个token的字节降序
                if self.bytes1 != other.bytes1:
                    return self.bytes1 > other.bytes1
                # 第一个token相同时，按第二个token的字节降序
                return self.bytes2 > other.bytes2
            
            def __eq__(self, other):
                return (self.count == other.count and 
                        self.bytes1 == other.bytes1 and 
                        self.bytes2 == other.bytes2)
            
            def get_pair(self):
                return (self.token_id1, self.token_id2)

        assert self.vocab_size >= len(self.stoi)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if self.special_tokens:  # Special Token
            special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens)})"
            text_parts = re.split(special_pattern, text)
        else:
            text_parts = [text]

        # Pre-Tokenizer
        initial_vocab_map = {v: k for k, v in self.itos.items()}
        token_groups = []
        for part in text_parts:
            if part in self.special_tokens or not part:
                continue
            words_in_bytes = pretokenize(part)
            for word in words_in_bytes:
                token_groups.append([initial_vocab_map[bytes([b])] for b in word])

        # BPE Merge
        idx = 0
        pair_counts = {}
        token = {}
        pre = {}
        nxt = {}
        pos = {}

        for i, token_lst in enumerate(token_groups):
            if not token_lst or len(token_lst) <= 1:
                continue
            token_lst_len = len(token_lst)
            for j, token_id in enumerate(token_lst):
                idx += 1
                token[idx] = token_id
                nxt[idx] = None if j == token_lst_len - 1 else idx + 1
                pre[idx] = None if j == 0 else idx - 1
                if j == token_lst_len - 1:
                    continue
                token_pair = (token_id, token_lst[j + 1])
                pair_counts[token_pair] = pair_counts.get(token_pair, 0) + 1
                if pos.get(token_pair) is None:
                    pos[token_pair] = set()
                pos[token_pair].add(idx)

        heap = []
        for (a, b), cnt in pair_counts.items():
            item = PairItem(cnt, a, b, self.itos)
            heapq.heappush(heap, item)

        def update_pair(pair: tuple[int, int], delta: int, pos_idx: int | None = None):
            if pair is None or None in pair: 
                return
            pair_counts[pair] = pair_counts.get(pair, 0) + delta
            cnt = pair_counts[pair]
            if cnt <= 0:
                pair_counts.pop(pair, None)
                pos.pop(pair, None)
                return
            if pos_idx is not None:
                ds = pos.setdefault(pair, set())
                if delta > 0:
                    ds.add(pos_idx)
                elif delta < 0:
                    ds.discard(pos_idx)
            a, b = pair
            item = PairItem(cnt, a, b, self.itos)
            heapq.heappush(heap, item)

        num_merges_needed = self.vocab_size - len(self.stoi)
        while num_merges_needed > 0 and heap:
            if not pair_counts: 
                break
            num_merges_needed -= 1
            
            while heap:
                item = heapq.heappop(heap)
                p1, p2 = item.get_pair()
                
                # 检查这个 pair 是否仍然有效
                if (p1, p2) not in pair_counts or pair_counts[(p1, p2)] != item.count:
                    continue  # 已经被合并过了

                # merge the new token
                self.merges.append((self.itos[p1], self.itos[p2]))

                p1_bytes, p2_bytes = self.itos[p1], self.itos[p2]
                new_token_bytes = p1_bytes + p2_bytes
                new_token_id = (
                    len(self.stoi)
                    if self.stoi.get(new_token_bytes) is None
                    else self.stoi[new_token_bytes]
                )
                self.stoi[new_token_bytes] = new_token_id
                self.itos[new_token_id] = new_token_bytes

                pos_lst = list(pos.get((p1, p2), set()))
                # modify the token group
                for pos_idx in pos_lst:
                    pre_idx = pre[pos_idx]
                    nxt_idx = nxt[pos_idx]
                    nnxt_idx = nxt[nxt_idx] if nxt_idx is not None else None

                    if nxt_idx is None or token[pos_idx] != p1 or token[nxt_idx] != p2: 
                        continue

                    if pre_idx is not None:
                        nxt[pre_idx] = pos_idx  # keep unchanged
                        update_pair((token[pre_idx], token[pos_idx]), -1, pre_idx)
                        update_pair((token[pre_idx], new_token_id), 1, pre_idx)
                    
                    if nnxt_idx is not None:
                        pre[nnxt_idx] = pos_idx
                        update_pair((token[nxt_idx], token[nnxt_idx]), -1, nxt_idx)
                        update_pair((new_token_id, token[nnxt_idx]), 1, pos_idx)
                    
                    pre[pos_idx] = pre_idx
                    nxt[pos_idx] = nnxt_idx
                    token[pos_idx] = new_token_id
                    token[nxt_idx] = None  # remove the old token
                    pre[nxt_idx] = None
                    nxt[nxt_idx] = None
                    
                pair_counts.pop((p1, p2), None)
                pos.pop((p1, p2), None)
                break

        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = self.itos.copy()
        self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in self.merges}


    def fast_train(self, path: str | os.PathLike):
        def bytes_desc(b):
            return bytes(255 - x for x in b)

        def pair_desc(pair):
            a = self.itos[pair[0]]
            b = self.itos[pair[1]]
            max_len = 2
            a_pad = a + bytes([0] * (max_len - len(a)))
            b_pad = b + bytes([0] * (max_len - len(b)))
            return (bytes_desc(a_pad), bytes_desc(b_pad))



        assert self.vocab_size >= len(self.stoi)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if self.special_tokens:  # Special Token
            special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens)})"
            text_parts = re.split(special_pattern, text)
        else:
            text_parts = [text]

        # Pre-Tokenizer
        initial_vocab_map = {v: k for k, v in self.itos.items()}
        token_groups = []
        for part in text_parts:
            if part in self.special_tokens or not part:
                continue
            words_in_bytes = pretokenize(part)
            for word in words_in_bytes:
                token_groups.append([initial_vocab_map[bytes([b])] for b in word])

        # BPE Merge
        idx = 0
        pair_counts = {}
        token = {}
        pre = {}
        nxt = {}
        pos = {}

        for i, token_lst in enumerate(token_groups):
            if not token_lst or len(token_lst) <= 1:
                continue
            token_lst_len = len(token_lst)
            for j, token_id in enumerate(token_lst):
                idx += 1
                token[idx] = token_id
                nxt[idx] = None if j == token_lst_len - 1 else idx + 1
                pre[idx] = None if j == 0 else idx - 1
                if j == token_lst_len - 1:
                    continue
                token_pair = (token_id, token_lst[j + 1])
                pair_counts[token_pair] = pair_counts.get(token_pair, 0) + 1
                if pos.get(token_pair) is None:
                    pos[token_pair] = set()
                pos[token_pair].add(idx)

        heap = [
            (
                -cnt,  # 频次取负，freq 高 → 数值小
                pair_desc((a, b)),
                a, b,
            )  # token-1 id, token-2 id
            for (a, b), cnt in pair_counts.items()
        ]
        heapq.heapify(heap)

        def update_pair(pair: tuple[int, int], delta: int, pos_idx: int | None = None):
            if pair is None or None in pair: return
            pair_counts[pair] = pair_counts.get(pair, 0) + delta
            cnt = pair_counts[pair]
            if cnt <= 0:
                pair_counts.pop(pair, None)
                pos.pop(pair, None)
                return
            if pos_idx is not None:
                ds = pos.setdefault(pair, set())
                if delta > 0:
                    ds.add(pos_idx)
                elif delta < 0:
                    ds.discard(pos_idx)
            a, b = pair
            heapq.heappush(heap, (-cnt, pair_desc((a, b)), a, b))

        num_merges_needed = self.vocab_size - len(self.stoi)
        while num_merges_needed > 0 and heap and len(heap) > 0:
            if not pair_counts: break
            num_merges_needed -= 1
            while heap and len(heap) > 0:
                neg_cnt, _, p1, p2 = heapq.heappop(heap)
                cnt = -neg_cnt
                if (p1, p2) not in pair_counts or pair_counts[(p1, p2)] != cnt:
                    continue  # 已经被合并过了

                # merge the new token
                self.merges.append((self.itos[p1], self.itos[p2]))

                p1_bytes, p2_bytes = self.itos[p1], self.itos[p2]
                new_token_bytes = p1_bytes + p2_bytes
                new_token_id = (
                    len(self.stoi)
                    if self.stoi.get(new_token_bytes) is None
                    else self.stoi[new_token_bytes]
                )
                self.stoi[new_token_bytes] = new_token_id
                self.itos[new_token_id] = new_token_bytes

                pos_lst = list(pos.get((p1, p2), set()))
                # modify the token group
                for pos_idx in pos_lst:
                    pre_idx = pre[pos_idx]
                    nxt_idx = nxt[pos_idx]
                    nnxt_idx = nxt[nxt_idx] if nxt_idx is not None else None

                    if nxt_idx is None or token[pos_idx] != p1 or token[nxt_idx] != p2: 
                        continue

                    if pre_idx is not None:
                        nxt[pre_idx] = pos_idx # keep uncanged
                        update_pair((token[pre_idx], token[pos_idx]), -1, pre_idx)
                        update_pair((token[pre_idx], new_token_id), 1, pre_idx)
                    
                    if nnxt_idx is not None:
                        pre[nnxt_idx] = pos_idx
                        update_pair((token[nxt_idx], token[nnxt_idx]), -1, nxt_idx)
                        update_pair((new_token_id, token[nnxt_idx]), 1, pos_idx)
                    
                    pre[pos_idx] = pre_idx
                    nxt[pos_idx] = nnxt_idx
                    token[pos_idx] = new_token_id
                    token[nxt_idx] = None  # remove the old token
                    pre[nxt_idx] = None
                    nxt[nxt_idx] = None
                    
                pair_counts.pop((p1, p2), None)
                pos.pop((p1, p2), None)
                break

        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = self.itos.copy()
        self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in self.merges}

    def slow_train(self, path: str | os.PathLike):
        assert self.vocab_size >= len(self.stoi)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if self.special_tokens:  # Special Token
            special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens)})"
            text_parts = re.split(special_pattern, text)
        else:
            text_parts = [text]

        initial_vocab_map = {v: k for k, v in self.itos.items()}
        token_groups = []
        for part in text_parts:
            if part in self.special_tokens or not part:
                continue
            words_in_bytes = pretokenize(part)
            for word in words_in_bytes:
                token_groups.append([initial_vocab_map[bytes([b])] for b in word])

        num_merges_needed = self.vocab_size - len(self.stoi)
        for i in range(num_merges_needed):
            pair_counts = self._get_stats(token_groups)
            if not pair_counts:
                break

            best_pair = max(
                pair_counts,
                key=lambda p: (pair_counts[p], self.itos[p[0]], self.itos[p[1]]),
            )

            new_token_id = len(self.itos)
            p1_bytes, p2_bytes = self.itos[best_pair[0]], self.itos[best_pair[1]]
            new_token_bytes = p1_bytes + p2_bytes

            self.merges.append((p1_bytes, p2_bytes))
            self.stoi[new_token_bytes] = new_token_id
            self.itos[new_token_id] = new_token_bytes

            token_groups = self._merge_pair_in_groups(
                token_groups, best_pair, new_token_id
            )

        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = self.itos.copy()
        self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in self.merges}

    def _get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        """Help encode"""
        return set(zip(tokens, tokens[1:]))

    def _encode_ordinary_text(self, text_bytes: bytes) -> list[int]:
        """BPE encode (不含特殊 token) —— 无额外列表 / O(n) 内存"""
        if not text_bytes:
            return []

        # ➊ 只解一次字节 → str
        try:
            text = text_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = text_bytes.decode("utf-8", errors="replace")

        ids_out = array("H")  # uint16 足够 ≤ 65k vocab

        pair_rank = self.merges_rank
        pair2new = self.pair2new
        byte2id = self.stoi  # 局部 alias，加速

        # ➋ 逐个“词块”处理，避免一次性 list
        for word_b in iter_pretokenize(text):
            # a. 初始：单字节 ids
            token_ids = array("H", (byte2id[bytes([b])] for b in word_b))

            # b. 就地合并：最经典 “greedy smallest-rank merge until稳定”
            while True:
                best_rank = 1000000000
                best_pos = -1
                # ——— 找当前序列里 rank 最小的 pair ———
                for i in range(len(token_ids) - 1):
                    r = pair_rank.get(
                        (self.itos[token_ids[i]], self.itos[token_ids[i + 1]]),
                        1000000000,
                    )
                    if r < best_rank:
                        best_rank, best_pos = r, i
                if best_pos == -1:
                    break
                # ——— 替换 best_pos & best_pos+1 为新的 token ———
                new_id = pair2new[
                    (self.itos[token_ids[best_pos]], self.itos[token_ids[best_pos + 1]])
                ]
                token_ids[best_pos : best_pos + 2] = array("H", [new_id])

            ids_out.extend(token_ids)

        # ➌ array → Python list（评测期望 list）
        return ids_out.tolist()

    def encode(self, text: str) -> list[int]:
        """Encode str"""
        if not text:
            return []

        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        if not sorted_special_tokens:
            return self._encode_ordinary_text(text.encode("utf-8"))

        special_pattern = f"({'|'.join(re.escape(s) for s in sorted_special_tokens)})"
        text_parts = re.split(special_pattern, text)

        all_ids = []
        for part in text_parts:
            if part in self.special_tokens:
                all_ids.append(self.stoi[part.encode("utf-8")])
            elif part:
                all_ids.extend(self._encode_ordinary_text(part.encode("utf-8")))
        return all_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
        *,
        output_format: str = "flat",
    ) -> Iterator[int] | Iterator[list[int]]:
        flat = output_format == "flat"
        for line in iterable:
            # —— 不要 strip 换行 ——          ▼
            ids = self.encode(line)
            if flat:
                yield from ids
            else:
                yield ids

    def decode(self, ids: list[int]) -> str:
        """ID -> text"""
        all_bytes = b"".join(self.itos.get(id, b"") for id in ids)
        return all_bytes.decode("utf-8", errors="replace")

    @classmethod
    def from_serialized(
        cls,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str],
    ):
        instance = cls(vocab_size=len(vocab), special_tokens=special_tokens)
        instance.stoi = {v: k for k, v in vocab.items()}
        instance.itos = vocab
        instance.merges = merges
        instance.merges_rank = {pair: i for i, pair in enumerate(merges)}
        instance.vocab = vocab

        instance.pair2new = {(p1, p2): instance.stoi[p1 + p2] for (p1, p2) in merges}

        return instance
