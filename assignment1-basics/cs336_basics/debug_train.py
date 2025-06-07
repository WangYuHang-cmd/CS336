# cs336_basics/debug_train.py

import sys
import os
from pathlib import Path
import json
import re  # 确保导入 re

# -- 设置路径以导入您的代码 ---
sys.path.insert(0, str(Path(__file__).parent.parent))
# --------------------------

# 从您的最终代码文件中导入 BPETokenizer 和 pretokenize
from cs336_basics.tokenizer import BPETokenizer, pretokenize 
from tests.common import gpt2_bytes_to_unicode, FIXTURES_PATH

def load_reference_merges(path: os.PathLike) -> list[tuple[bytes, bytes]]:
    """从参考文件中加载标准合并规则。"""
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(path) as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    return reference_merges

class DebugBPETokenizer(BPETokenizer):
    """
    一个继承自您的分词器的调试专用类。
    我们只重写 train 方法以加入逐行调试逻辑。
    """
    def train(self, path: str | os.PathLike, reference_merges: list[tuple[bytes, bytes]]):
        # --- 这部分逻辑完全复制自您最新版本的 train 方法 ---
        assert self.vocab_size >= len(self.stoi)
        
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if self.special_tokens:
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

        # 初始计频
        pair_counts = {}
        for group in token_groups:
            for pair in zip(group, group[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        num_merges_needed = self.vocab_size - len(self.stoi)
        for i in range(num_merges_needed):
            if not pair_counts:
                print("--- No more pairs to merge. Stopping. ---")
                break
            
            # 找到最佳对
            my_best_pair_ids = max(pair_counts, key=lambda p: (pair_counts[p], (-p[0], -p[1])))
            
            # --- 核心调试逻辑 ---
            my_choice_bytes = (self.itos[my_best_pair_ids[0]], self.itos[my_best_pair_ids[1]])
            reference_choice_bytes = reference_merges[i]
            
            if my_choice_bytes != reference_choice_bytes:
                print("="*80)
                print(f"!!! DIVERGENCE FOUND AT MERGE #{i} !!!")
                print("="*80)
                
                print(f"\n[YOUR ALGORITHM'S CHOICE]")
                print(f"Pair: {my_choice_bytes!r}")
                print(f"Frequency in your counts: {pair_counts.get(my_best_pair_ids, 'N/A')}")
                
                print(f"\n[CORRECT CHOICE FROM REFERENCE]")
                print(f"Pair: {reference_choice_bytes!r}")
                
                try:
                    # 将参考的bytes对转换回ID
                    # 注意：如果参考对中的某个token还没在您的词汇表中，这里会出错
                    ref_p1_id = self.stoi.get(reference_choice_bytes[0])
                    ref_p2_id = self.stoi.get(reference_choice_bytes[1])
                    if ref_p1_id is not None and ref_p2_id is not None:
                        ref_pair_ids = (ref_p1_id, ref_p2_id)
                        print(f"Frequency in your counts: {pair_counts.get(ref_pair_ids, 'This pair was not found in your counts dict')}")
                    else:
                        print("Frequency in your counts: Correct token part not found in your `stoi` map.")
                except Exception as e:
                    print(f"Could not analyze reference pair frequency due to error: {e}")

                print("\n[CONTEXT: Top 15 Most Frequent Pairs in Your Algorithm at This Step]")
                sorted_counts = sorted(pair_counts.items(), key=lambda item: (item[1], (-item[0][0], -item[0][1])), reverse=True)
                for rank, (pair, count) in enumerate(sorted_counts[:15]):
                    pair_bytes = (self.itos.get(pair[0]), self.itos.get(pair[1]))
                    is_your_choice = "  <-- YOUR CHOICE" if pair == my_best_pair_ids else ""
                    print(f"  {rank+1}. Pair: {pair_bytes!r}, Freq: {count}{is_your_choice}")
                
                print("\n--- DEBUGGING STOPPED ---")
                raise AssertionError("Training path diverged from reference.")
            # --- 调试逻辑结束 ---

            new_token_id = len(self.special_tokens) + 256 + i
            self.merges.append(my_choice_bytes)
            self.stoi[my_choice_bytes[0] + my_choice_bytes[1]] = new_token_id
            self.itos[new_token_id] = my_choice_bytes[0] + my_choice_bytes[1]
            
            # 原地更新逻辑（从您的代码中复制）
            new_token_groups = []
            for group in token_groups:
                new_group = []
                j = 0
                while j < len(group):
                    if j < len(group) - 1 and (group[j], group[j+1]) == my_best_pair_ids:
                        new_group.append(new_token_id)
                        if j > 0:
                            old_left_neighbor = (group[j-1], my_best_pair_ids[0])
                            pair_counts[old_left_neighbor] = pair_counts.get(old_left_neighbor, 0) - 1
                            new_left_neighbor = (group[j-1], new_token_id)
                            pair_counts[new_left_neighbor] = pair_counts.get(new_left_neighbor, 0) + 1
                        if j < len(group) - 2:
                            old_right_neighbor = (my_best_pair_ids[1], group[j+2])
                            pair_counts[old_right_neighbor] = pair_counts.get(old_right_neighbor, 0) - 1
                            new_right_neighbor = (new_token_id, group[j+2])
                            pair_counts[new_right_neighbor] = pair_counts.get(new_right_neighbor, 0) + 1
                        j += 2
                    else:
                        new_group.append(group[j])
                        j += 1
                new_token_groups.append(new_group)
            token_groups = new_token_groups
            del pair_counts[my_best_pair_ids]

        self.vocab = self.itos.copy()
        
        print("\n--- Congratulations! Your training process perfectly matched the reference for all merges! ---")

if __name__ == "__main__":
    print("--- Starting BPE Training Debugger ---")
    
    input_path = FIXTURES_PATH / "corpus.en"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"
    print(f"Loading reference merges from: {reference_merges_path}")
    reference_merges = load_reference_merges(reference_merges_path)
    
    print(f"Initializing tokenizer with vocab_size={vocab_size} and special_tokens={special_tokens}")
    debug_tokenizer = DebugBPETokenizer(vocab_size=vocab_size, special_tokens=special_tokens)
    
    try:
        debug_tokenizer.train(input_path, reference_merges)
    except AssertionError as e:
        print(f"\nError: {e}")