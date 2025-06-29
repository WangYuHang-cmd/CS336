# cs336_basics/tokenizer.py

from typing import Iterable, Iterator, List, Dict, Tuple
import os
import regex as re
from array import array



GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(text: str) -> list[bytes]:
    """使用GPT-2的正则表达式将文本分割成“词块”，并编码为bytes。 This step is very important!!!! Otherwise the b'a\n\nb' will be transfer into 'a' '\n\n' 'b' instead of 'a' '\n' '\n' 'b'  """
    str_tokens = re.findall(GPT2_SPLIT_PATTERN, text)
    byte_tokens = [s.encode('utf-8') for s in str_tokens]
    return byte_tokens


GPT2_RE = re.compile(GPT2_SPLIT_PATTERN)
def iter_pretokenize(text: str) -> Iterator[bytes]:
    """按 GPT-2 正则逐个产生字节串，零内存列表。"""
    for m in GPT2_RE.finditer(text):
        yield m.group(0).encode('utf-8')



class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str] | None = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [token.encode("utf-8") for token in self.special_tokens]
        
        self.merges: List[Tuple[bytes, bytes]] = []
        self.stoi: Dict[bytes, int] = {}
        self.itos: Dict[int, bytes] = {}
        self.merges_rank: Dict[Tuple[bytes, bytes], int] = {}
        
        # init vocab
        for i, token_bytes in enumerate(self.special_tokens_bytes): # special tokens
            self.stoi[token_bytes] = i
            self.itos[i] = token_bytes
        
        
        offset = len(self.special_tokens_bytes) # 单字节 tokens
        for i in range(256):
            self.stoi[bytes([i])] = i + offset
            self.itos[i + offset] = bytes([i])
        
        self.vocab = self.itos.copy() # for serialization
        self.merges_rank = {} # for fast lookup
        # pair2new: (p1, p2) -> new_token_id
        self.pair2new = {
            (p1, p2): self.stoi[p1+p2]
                for (p1,p2) in self.merges
        }

    def _get_stats(self, token_groups: list[list[int]]):
        """Count the frequency of occurrence of all byte pairs."""
        pair_counts = {}
        for group in token_groups:
            for pair in zip(group, group[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    def _merge_pair_in_groups(self, ids_group: list[list[int]], pair_to_merge: tuple[int, int], new_id: int):
        """One merge in vocab"""
        new_ids_group = []
        for group in ids_group:
            new_group = []
            i = 0
            while i < len(group):
                if i < len(group) - 1 and (group[i], group[i+1]) == pair_to_merge:
                    new_group.append(new_id)
                    i += 2
                else:
                    new_group.append(group[i])
                    i += 1
            new_ids_group.append(new_group)
        return new_ids_group

    def train(self, path: str | os.PathLike):
        assert self.vocab_size >= len(self.stoi)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if self.special_tokens: # Special Token
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
        num_merges_needed = self.vocab_size - len(self.stoi)
        for i in range(num_merges_needed):
            pair_counts = self._get_stats(token_groups)
            if not pair_counts:
                break

            # a. 【关键】找到频率最高的对，使用与参考实现一致的平局决胜规则
            # 规则：1. 按频率降序; 2. 按第一个token的解码字符串降序; 3. 按第二个token的解码字符串降序
            best_pair = max(pair_counts, key=lambda p: (
                pair_counts[p],
                self.itos[p[0]],
                self.itos[p[1]]
            ))

            # b. Nwe token
            new_token_id = len(self.itos)
            p1_bytes, p2_bytes = self.itos[best_pair[0]], self.itos[best_pair[1]]
            new_token_bytes = p1_bytes + p2_bytes
            
            self.merges.append((p1_bytes, p2_bytes))
            self.stoi[new_token_bytes] = new_token_id
            self.itos[new_token_id] = new_token_bytes

            # c. Rebuld vocab
            token_groups = self._merge_pair_in_groups(token_groups, best_pair, new_token_id)

        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = self.itos.copy()
        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in self.merges}
        
    
    def _get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        """Help encode"""
        return set(zip(tokens, tokens[1:]))

    from array import array

    def _encode_ordinary_text(self, text_bytes: bytes) -> list[int]:
        """BPE encode (不含特殊 token) —— 无额外列表 / O(n) 内存"""
        if not text_bytes:
            return []

        # ➊ 只解一次字节 → str
        try:
            text = text_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = text_bytes.decode('utf-8', errors='replace')

        ids_out = array('H')                      # uint16 足够 ≤ 65k vocab

        pair_rank  = self.merges_rank
        pair2new   = self.pair2new
        byte2id    = self.stoi                     # 局部 alias，加速

        # ➋ 逐个“词块”处理，避免一次性 list
        for word_b in iter_pretokenize(text):
            # a. 初始：单字节 ids
            token_ids = array('H', (byte2id[bytes([b])] for b in word_b))

            # b. 就地合并：最经典 “greedy smallest-rank merge until稳定”
            while True:
                best_rank = 1_000_000_000
                best_pos  = -1
                # ——— 找当前序列里 rank 最小的 pair ———
                for i in range(len(token_ids) - 1):
                    r = pair_rank.get((self.itos[token_ids[i]], self.itos[token_ids[i+1]]), 1_000_000_000)
                    if r < best_rank:
                        best_rank, best_pos = r, i
                if best_pos == -1:
                    break
                # ——— 替换 best_pos & best_pos+1 为新的 token ———
                new_id = pair2new[(self.itos[token_ids[best_pos]], self.itos[token_ids[best_pos+1]])]
                token_ids[best_pos:best_pos+2] = array('H', [new_id])

            ids_out.extend(token_ids)

        # ➌ array → Python list（评测期望 list）
        return ids_out.tolist()

    # def _encode_ordinary_text(self, text_bytes: bytes) -> list[int]:
    #     """BPE encode except special tokens"""
    #     if not text_bytes: return []
        
    #     try:
    #         text = text_bytes.decode('utf-8')
    #     except UnicodeDecodeError:
    #         text = text_bytes.decode('utf-8', errors='replace')
        
    #     words_in_bytes = pretokenize(text)
    #     all_ids = []
    #     for word_bytes in words_in_bytes:
    #         if not word_bytes: continue
            
    #         tokens = [bytes([b]) for b in word_bytes]
    #         if len(tokens) <= 1:
    #             all_ids.extend([self.stoi.get(t) for t in tokens if t in self.stoi])
    #             continue
                
    #         while True:
    #             pairs = self._get_pairs(tokens)
    #             best_pair = min(
    #                 (p for p in pairs if p in self.merges_rank),
    #                 key=lambda p: self.merges_rank[p],
    #                 default=None
    #             )
    #             if best_pair is None: break
                
    #             left, right = best_pair
    #             new_tokens = []
    #             i = 0
    #             while i < len(tokens):
    #                 if i < len(tokens) - 1 and tokens[i] == left and tokens[i+1] == right:
    #                     new_tokens.append(left + right)
    #                     i += 2
    #                 else:
    #                     new_tokens.append(tokens[i])
    #                     i += 1
    #             tokens = new_tokens
                
    #         all_ids.extend([self.stoi[t] for t in tokens])
    #     return all_ids

    def encode(self, text: str) -> list[int]:
        """Encode str"""
        if not text: return []
        
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        if not sorted_special_tokens:
            return self._encode_ordinary_text(text.encode('utf-8'))

        special_pattern = f"({'|'.join(re.escape(s) for s in sorted_special_tokens)})"
        text_parts = re.split(special_pattern, text)
        
        all_ids = []
        for part in text_parts:
            if part in self.special_tokens:
                all_ids.append(self.stoi[part.encode('utf-8')])
            elif part:
                all_ids.extend(self._encode_ordinary_text(part.encode('utf-8')))
        return all_ids

    # def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    #     """Encode line"""
    #     for line in iterable:
    #         yield from self.encode(line)
    
    def encode_iterable(
        self,
        iterable: Iterable[str],
        *,
        output_format: str = "flat",
    ) -> Iterator[int] | Iterator[list[int]]:
        flat = (output_format == "flat")
        for line in iterable:
            # —— 不要 strip 换行 ——          ▼
            ids = self.encode(line)
            if flat:
                yield from ids
            else:
                yield ids




    def decode(self, ids: list[int]) -> str:
        """ID -> text"""
        all_bytes = b"".join(self.itos.get(id, b'') for id in ids)
        return all_bytes.decode("utf-8", errors="replace")

    # @classmethod
    # def from_serialized(cls, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
    #     """Build Tokenizer"""
    #     instance = cls(vocab_size=len(vocab), special_tokens=special_tokens)
    #     instance.stoi = {v: k for k, v in vocab.items()}
    #     instance.itos = vocab
    #     instance.merges = merges
    #     instance.merges_rank = {pair: i for i, pair in enumerate(merges)}
    #     instance.vocab = vocab
    #     return instance
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

        # ★ 重新构建 fast-lookup 映射 ★
        instance.pair2new = {
            (p1, p2): instance.stoi[p1 + p2] for (p1, p2) in merges
        }

        return instance
