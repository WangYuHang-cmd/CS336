import os
from .utils import *


class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str], **kwargs):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [
            token.encode("utf-8") for token in self.special_tokens
        ]
        self.merges: list[tuple[bytes, bytes]] = []
        self.stoi, self.itos = {}, {}
        for i, stok in enumerate(self.special_tokens):
            self.stoi[stok.encode("utf-8")] = i
            self.itos[i] = stok.encode("utf-8")

        soffset = len(self.special_tokens)
        for b in range(256):
            self.stoi[bytes([b])] = b + soffset
            self.itos[b + soffset] = bytes([b])
        self._corpus_ids: list[int] = []

        # self.root = Trie()

    def _id(self, b: bytes):
        id = self.stoi.get(b)
        return id

    def _bytes(self, i: int):
        output = self.itos.get(i)
        return output

    def train(self, path: str | os.PathLike):
        chunk_size = 1 << 18
        overlap_size = (
            max(len(tokens) for tokens in self.special_tokens_bytes) - 1
            if self.special_tokens_bytes
            else 0
        )

        # Process all the text into tokens into _corpus_ids
        with open(path, "rb") as file:
            pre_chunk = b""

            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    for each_byte in pre_chunk:
                        self._corpus_ids.append(self._id(bytes([each_byte])))
                    break

                pre_chunk = pre_chunk + chunk

                while True and len(self.special_tokens):
                    pos = len(pre_chunk)
                    if pos == 0:
                        break
                    s_tok = b""
                    for special_token in self.special_tokens_bytes:
                        tpos = pre_chunk.find(special_token)
                        if tpos != -1 and tpos < pos:
                            pos = tpos
                            s_tok = special_token
                    if pos == len(pre_chunk):
                        break

                    # xxx[pos...]xxxx
                    for i in range(0, pos):
                        self._corpus_ids.append(self._id(bytes([pre_chunk[i]])))
                    self._corpus_ids.append(self._id(s_tok))
                    pre_chunk = pre_chunk[pos + len(s_tok) :]

                for each_byte in pre_chunk[:-overlap_size]:
                    self._corpus_ids.append(self._id(bytes([each_byte])))
                pre_chunk = pre_chunk[-overlap_size:]

        current_len = len(self.stoi)

        while current_len < self.vocab_size:
            couple_sum = {}
            max_sum = 0
            max_couple = (None, None)

            vocab_len = len(self._corpus_ids)
            for i in range(vocab_len - 1):
                if (
                    self.itos[self._corpus_ids[i]] in self.special_tokens_bytes
                    or self.itos[self._corpus_ids[i + 1]] in self.special_tokens_bytes
                ):
                    continue
                new_couple = (
                    self.itos[self._corpus_ids[i]],
                    self.itos[self._corpus_ids[i + 1]],
                )
                if couple_sum.get(new_couple):
                    couple_sum[new_couple] += 1
                else:
                    couple_sum[new_couple] = 1
                if couple_sum[new_couple] > max_sum:
                    max_sum = couple_sum[new_couple]
                    max_couple = new_couple

            if max_sum == 0 or (None in max_couple):
                break
            new_tok = new_couple[0] + new_couple[1]
            self.merges.append(new_couple)
            self.stoi[new_tok] = current_len
            self.itos[current_len] = new_tok
            current_len += 1

            old_corpus_ids = self._corpus_ids[:]
            self._corpus_ids = []

            idx = 0
            while idx < vocab_len:
                if idx == vocab_len - 1:
                    self._corpus_ids.append(old_corpus_ids[idx])
                    idx += 1
                    continue
                cur_tok = (
                    self.itos[old_corpus_ids[idx]] + self.itos[old_corpus_ids[idx + 1]]
                )
                if cur_tok == new_tok:
                    self._corpus_ids.append(self.stoi[cur_tok])
                    idx += 1
                else:
                    self._corpus_ids.append(old_corpus_ids[idx])
                idx += 1

        # Build a Trie to help encode
        # self.build_Trie()

    def build_Trie(self):
        root = Trie()
        for token, token_id in self.stoi.items():
            if token in self.special_tokens_bytes:
                continue
            cur_node = root
            token_len = len(token)
            for i in range(token_len):
                cur_char = bytes([token[i]])
                if cur_node.children.get(cur_char) is None:
                    cur_node.children[cur_char] = Trie()
                cur_node = cur_node.children[cur_char]
            cur_node.token_id = token_id
        self.root = root

    def find_token(self, text: bytes):
        token = b""
        if len(text) == 0:
            return token
        cur_node = self.root
        for each_int in text:
            each_byte = bytes([each_int])
            if cur_node.children.get(each_byte) is None:
                break
            cur_node = cur_node.children.get(each_byte)
            if cur_node.token_id is not None:
                token = self.itos[cur_node.token_id]
        if len(token) == 0:
            return bytes([text[0]])
        return token

    # def encode(self, text: str) -> list[int]:
    #     ids: list[int] = []
    #     text_bytes = text.encode("utf-8")
    #     idx = 0
    #     text_len = len(text_bytes)
    #     while idx < text_len:
    #         s_tok = b""
    #         max_len = 0
    #         pos = -1
    #         for special_token in self.special_tokens_bytes:
    #             n_pos = text_bytes.find(special_token, idx)
    #             if n_pos == -1:
    #                 continue
    #             if pos == -1:
    #                 pos, s_tok, max_len = n_pos, special_token, len(special_token)
    #             elif n_pos < pos:
    #                 pos, s_tok, max_len = n_pos, special_token, len(special_token)
    #             elif n_pos == pos and len(special_token) > max_len:
    #                 pos, s_tok, max_len = n_pos, special_token, len(special_token)

    #         if pos == idx:
    #             ids.append(self.stoi[s_tok])
    #             idx += len(s_tok)
    #             continue

    #         if idx >= text_len:
    #             continue

    #         find_text = text_bytes[idx:pos] if pos != -1 else text_bytes[idx:]
    #         token = self.find_token(find_text)
    #         ids.append(self.stoi[token])
    #         idx += len(token)

    #     return ids
    
    # ===========================================================================================================
    def encode(self, text: str) -> list[int]:
        """
        编码文本为token IDs
        """
        if not text:
            return []
        
        text_bytes = text.encode("utf-8")
        ids = []
        current_pos = 0
        
        while current_pos < len(text_bytes):
            # 检查是否有特殊token在当前位置
            matched_special = None
            
            if self.special_tokens_bytes:
                # 按长度降序检查，确保最长匹配
                for special_token in sorted(self.special_tokens_bytes, key=len, reverse=True):
                    end_pos = current_pos + len(special_token)
                    if end_pos <= len(text_bytes) and text_bytes[current_pos:end_pos] == special_token:
                        matched_special = special_token
                        break
            
            if matched_special:
                ids.append(self.stoi[matched_special])
                current_pos += len(matched_special)
                continue
            
            # 找到下一个特殊token的位置
            next_special_pos = len(text_bytes)
            if self.special_tokens_bytes:
                for special_token in self.special_tokens_bytes:
                    pos = text_bytes.find(special_token, current_pos)
                    if pos != -1 and pos < next_special_pos:
                        next_special_pos = pos
            
            # 处理普通文本段
            if current_pos < next_special_pos:
                chunk = text_bytes[current_pos:next_special_pos]
                chunk_ids = self._encode_ordinary_text(chunk)
                ids.extend(chunk_ids)
                current_pos = next_special_pos
        
        return ids

    def _encode_ordinary_text(self, text_bytes: bytes) -> list[int]:
        """
        对不包含特殊token的普通文本进行BPE编码
        重要：只使用通过merge规则生成的tokens！
        """
        if len(text_bytes) == 0:
            return []
        
        # 初始化为单字节tokens
        tokens = [bytes([b]) for b in text_bytes]
        
        # 按照合并规则的顺序进行合并
        for left_part, right_part in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == left_part and tokens[i + 1] == right_part:
                    # 执行合并
                    tokens[i] = left_part + right_part
                    tokens.pop(i + 1)
                    # 不增加i，因为新token可能继续合并
                else:
                    i += 1
        
        # 转换为token IDs
        # 重要：只使用单字节tokens或通过merge生成的tokens
        result = []
        for token in tokens:
            if token in self.stoi:
                result.append(self.stoi[token])
            else:
                # 如果token不在词汇表中，分解为单字节
                # 这种情况不应该发生，但作为安全措施
                for b in token:
                    result.append(self.stoi[bytes([b])])
        
        return result

    def encode_iterable(self, iterable):
        """
        Memory-efficient encoding - process in very small chunks
        """
        buffer = ""
        CHUNK_SIZE = 50  # 非常小的块
        
        for line in iterable:
            buffer += line
            
            # 处理完整的块
            while len(buffer) >= CHUNK_SIZE:
                # 找到一个安全的切分点，避免在特殊token中间切断
                safe_end = CHUNK_SIZE
                
                # 向前搜索，确保不会切断可能的特殊token
                if self.special_tokens:
                    max_special_len = max(len(st) for st in self.special_tokens)
                    # 检查是否可能在特殊token中间
                    for offset in range(min(max_special_len, CHUNK_SIZE)):
                        test_pos = CHUNK_SIZE - offset
                        if test_pos <= 0:
                            continue
                        # 检查这个位置是否安全
                        chunk_end = buffer[test_pos:test_pos + max_special_len]
                        if any(st.startswith(chunk_end) for st in self.special_tokens if len(st) > len(chunk_end)):
                            safe_end = test_pos
                            break
                
                if safe_end > 0:
                    chunk = buffer[:safe_end]
                    ids = self.encode(chunk)
                    for id in ids:
                        yield id
                    buffer = buffer[safe_end:]
                else:
                    # 如果找不到安全点，至少处理一个字符
                    ids = self.encode(buffer[0])
                    for id in ids:
                        yield id
                    buffer = buffer[1:]
        
        # 处理剩余的buffer
        if buffer:
            ids = self.encode(buffer)
            for id in ids:
                yield id
    # ===========================================================================================================
    

    def decode(self, ids: list[int]) -> str:
        ctx = ""
        pre = b""
        for id in ids:
            cur = self.itos[id]
            if len(pre):
                cur = pre + cur
                pre = b""
            try:
                ctx += cur.decode("utf-8")
            except:
                pre = cur
        return ctx

    @classmethod
    def from_serialized(
        cls,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str],
    ):
        # cls to a new tokenizer
        tokenizer = cls(vocab_size=len(vocab), special_tokens=special_tokens)
        tokenizer.stoi = {v: k for k, v in vocab.items()}
        tokenizer.itos = vocab.copy()
        tokenizer.merges = merges.copy()
        if special_tokens:
            tokenizer.special_token = [s for s in special_tokens]
            tokenizer.special_tokens_bytes = [s.encode("utf-8") for s in special_tokens]
        # tokenizer.build_Trie()
        return tokenizer
