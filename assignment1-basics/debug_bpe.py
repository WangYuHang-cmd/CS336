# # # debug_bpe.py
# # import pathlib, json, heapq, copy
# # from cs336_basics.tokenizer import BPETokenizer
# # from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

# # INPUT_PATH   = FIXTURES_PATH / "corpus.en"
# # VOCAB_SIZE   = 500
# # SPECIALS     = ["<|endoftext|>"]

# # def run_slow(path):
# #     tk = BPETokenizer(vocab_size=VOCAB_SIZE, special_tokens=SPECIALS)
# #     tk.slow_train(path)          # 你保留的「参考实现」函数
# #     return tk

# # def run_fast(path):
# #     tk = BPETokenizer(vocab_size=VOCAB_SIZE, special_tokens=SPECIALS)
# #     tk.train(path)               # 你的 fast_train
# #     return tk

# # def compare_merges(slow, fast, max_show=10):
# #     """
# #     打印第一次分叉及其后几步。slow == 参考（slow_train），
# #     fast == 你的 train。
# #     """
# #     for i, (slow_pair, fast_pair) in enumerate(zip(slow.merges, fast.merges), 1):
# #         if slow_pair != fast_pair:
# #             print(f"\n❌  第 {i} 次 merge 开始分叉")
# #             print(f"    slow : {slow_pair}")
# #             print(f"    fast : {fast_pair}")
# #             for j in range(i, i + max_show):
# #                 s = slow.merges[j - 1] if j <= len(slow.merges) else None
# #                 f = fast.merges[j - 1] if j <= len(fast.merges) else None
# #                 print(f"{j:>4}  slow={s}   fast={f}")
# #             return i
# #     print("✅ 两者 merge 序列完全一致")
# #     return None


# # def sanity_check_heap(tk_fast, source_text, steps=50):
# #     """确认 fast 算法每一步的堆顶 == naive max(pair_counts)"""
# #     # 拷贝出局部可变对象
# #     pair_counts = copy.deepcopy(tk_fast._get_stats(
# #         [[tk_fast.stoi[bytes([c])]] for c in source_text.encode("utf-8")]
# #     ))
# #     merges = []
# #     for step in range(steps):
# #         if not pair_counts: break
# #         max_pair = max(pair_counts, key=lambda p:(pair_counts[p], p[0], p[1]))
# #         # 从堆里找实际弹出的
# #         while tk_fast._heap:
# #             cnt_neg, t1, t2, p1, p2 = heapq.heappop(tk_fast._heap)
# #             if pair_counts.get((p1,p2),0) == -cnt_neg:
# #                 heap_pair = (p1,p2); break
# #         ok = heap_pair == max_pair
# #         print(f"{step+1:>3}: heap={heap_pair}  max={max_pair}   {'✅' if ok else '❌'}")
# #         if not ok: break
# #         # 假装合并（只更新 pair_counts），给下次循环用
# #         cnt = pair_counts.pop(max_pair)
# #         # 更新 pair_counts 中受影响的相邻 pair（慢但够调试）
# #         # …此处可省略，主要目的是看第一步是否错
# #         break

# # def main():
# #     slow = run_slow(INPUT_PATH)
# #     fast = run_fast(INPUT_PATH)
# #     compare_merges(slow, fast)

# #     # ---------- 如还需要验证堆顺序，取消下面两行注释 ----------
# #     # with open(INPUT_PATH, "r", encoding="utf-8") as f:
# #     #     sanity_check_heap(fast, f.read())

# # if __name__ == "__main__":
# #     main()

# # debug_bpe.py
# import pathlib, json, heapq, copy
# from cs336_basics.tokenizer import BPETokenizer
# from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

# INPUT_PATH   = FIXTURES_PATH / "corpus.en"
# VOCAB_SIZE   = 500
# SPECIALS     = ["<|endoftext|>"]

# def run_slow(path):
#     tk = BPETokenizer(vocab_size=VOCAB_SIZE, special_tokens=SPECIALS)
#     tk.slow_train(path)          # 你保留的「参考实现」函数
#     return tk

# def run_fast(path):
#     tk = BPETokenizer(vocab_size=VOCAB_SIZE, special_tokens=SPECIALS)
#     tk.train(path)               # 你的 fast_train
#     return tk

# def get_pair_stats_at_step(tk, path, step):
#     """重现第 step 步时的 pair_counts 状态"""
#     with open(path, "r", encoding="utf-8") as f:
#         text = f.read()
    
#     if tk.special_tokens:
#         special_pattern = f"({'|'.join(tk.special_tokens)})"
#         import re
#         text_parts = re.split(special_pattern, text)
#     else:
#         text_parts = [text]
    
#     # 重建初始 token groups
#     initial_vocab_map = {v: k for k, v in tk.itos.items() if k < 256}
#     token_groups = []
#     for part in text_parts:
#         if part in tk.special_tokens or not part:
#             continue
#         # 假设有 pretokenize 函数
#         words_in_bytes = tk.pretokenize(part) if hasattr(tk, 'pretokenize') else [part.encode('utf-8')]
#         for word in words_in_bytes:
#             token_groups.append([initial_vocab_map[bytes([b])] for b in word])
    
#     # 应用前 step-1 次合并
#     for i in range(min(step-1, len(tk.merges))):
#         p1_bytes, p2_bytes = tk.merges[i]
#         p1_id = tk.stoi[p1_bytes]
#         p2_id = tk.stoi[p2_bytes]
#         new_id = tk.stoi[p1_bytes + p2_bytes]
#         token_groups = tk._merge_pair_in_groups(token_groups, (p1_id, p2_id), new_id)
    
#     # 统计当前的 pair counts
#     return tk._get_stats(token_groups)

# def compare_merges_detailed(slow, fast, path, max_show=10):
#     """
#     详细比较合并过程，包括频数、token ID等信息
#     """
#     print("="*80)
#     print(f"比较 slow_train 和 train 的合并过程")
#     print(f"词表大小: {VOCAB_SIZE}, 特殊token: {SPECIALS}")
#     print("="*80)
    
#     # 检查初始词表是否一致
#     print("\n初始词表检查:")
#     print(f"slow 初始词表大小: {len([k for k in slow.itos if k < 256])}")
#     print(f"fast 初始词表大小: {len([k for k in fast.itos if k < 256])}")
    
#     # 逐步比较合并
#     for i in range(min(len(slow.merges), len(fast.merges))):
#         slow_pair = slow.merges[i]
#         fast_pair = fast.merges[i]
        
#         if slow_pair != fast_pair:
#             print(f"\n❌ 第 {i+1} 次 merge 开始分叉!")
#             print("-"*80)
            
#             # 获取这一步的 pair counts
#             slow_stats = get_pair_stats_at_step(slow, path, i+1)
#             fast_stats = get_pair_stats_at_step(fast, path, i+1)
            
#             # 详细显示分叉点
#             print(f"\n分叉详情:")
#             print(f"{'':>10} {'Pair':>30} {'Count':>10} {'Token IDs':>20} {'New Token ID':>15}")
#             print("-"*95)
            
#             # slow 选择的 pair
#             slow_p1_id = slow.stoi[slow_pair[0]]
#             slow_p2_id = slow.stoi[slow_pair[1]]
#             slow_new_id = slow.stoi[slow_pair[0] + slow_pair[1]]
#             slow_count = slow_stats.get((slow_p1_id, slow_p2_id), 0)
            
#             print(f"{'slow:':>10} {str(slow_pair):>30} {slow_count:>10} "
#                   f"{f'({slow_p1_id}, {slow_p2_id})':>20} {slow_new_id:>15}")
            
#             # fast 选择的 pair
#             fast_p1_id = fast.stoi[fast_pair[0]]
#             fast_p2_id = fast.stoi[fast_pair[1]]
#             fast_new_id = fast.stoi[fast_pair[0] + fast_pair[1]]
#             fast_count = fast_stats.get((fast_p1_id, fast_p2_id), 0)
            
#             print(f"{'fast:':>10} {str(fast_pair):>30} {fast_count:>10} "
#                   f"{f'({fast_p1_id}, {fast_p2_id})':>20} {fast_new_id:>15}")
            
#             # 显示频数相同的其他候选
#             print(f"\n频数为 {max(slow_count, fast_count)} 的所有候选 pairs:")
#             candidates = []
#             for (p1_id, p2_id), count in slow_stats.items():
#                 if count == max(slow_count, fast_count):
#                     p1_bytes = slow.itos[p1_id]
#                     p2_bytes = slow.itos[p2_id]
#                     candidates.append(((p1_bytes, p2_bytes), count, (p1_id, p2_id)))
            
#             # 按照两种排序规则排序
#             slow_sorted = sorted(candidates, key=lambda x: (x[1], x[0][0], x[0][1]))
#             fast_sorted = sorted(candidates, key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)
            
#             print(f"\nslow_train 排序 (升序):")
#             for j, (pair, count, ids) in enumerate(slow_sorted[-5:], 1):
#                 marker = "✓" if pair == slow_pair else " "
#                 print(f"  {marker} {j}. {str(pair):>30} count={count} ids={ids}")
            
#             print(f"\ntrain 排序 (降序):")
#             for j, (pair, count, ids) in enumerate(fast_sorted[:5], 1):
#                 marker = "✓" if pair == fast_pair else " "
#                 print(f"  {marker} {j}. {str(pair):>30} count={count} ids={ids}")
            
#             # 继续显示后续几步
#             print(f"\n后续 {max_show} 步合并:")
#             print(f"{'Step':>6} {'slow pair':>30} {'fast pair':>30} {'Match':>8}")
#             print("-"*80)
            
#             for j in range(i, min(i + max_show, min(len(slow.merges), len(fast.merges)))):
#                 s = slow.merges[j] if j < len(slow.merges) else None
#                 f = fast.merges[j] if j < len(fast.merges) else None
#                 match = "✓" if s == f else "✗"
#                 print(f"{j+1:>6} {str(s):>30} {str(f):>30} {match:>8}")
            
#             return i + 1
    
#     print("\n✅ 两个实现的合并序列完全一致!")
#     print(f"总合并次数: {len(slow.merges)}")
#     return None

# def check_token_reuse(slow, fast):
#     """检查 token ID 复用情况"""
#     print("\n" + "="*80)
#     print("Token ID 分配检查")
#     print("="*80)
    
#     # 检查是否有重复的 token bytes 对应不同 ID
#     slow_bytes_to_ids = {}
#     fast_bytes_to_ids = {}
    
#     for token_id, token_bytes in slow.itos.items():
#         if token_bytes not in slow_bytes_to_ids:
#             slow_bytes_to_ids[token_bytes] = []
#         slow_bytes_to_ids[token_bytes].append(token_id)
    
#     for token_id, token_bytes in fast.itos.items():
#         if token_bytes not in fast_bytes_to_ids:
#             fast_bytes_to_ids[token_bytes] = []
#         fast_bytes_to_ids[token_bytes].append(token_id)
    
#     # 找出有多个 ID 的 token
#     print("\nslow_train 中的重复 token:")
#     slow_dups = {k: v for k, v in slow_bytes_to_ids.items() if len(v) > 1}
#     if slow_dups:
#         for token_bytes, ids in list(slow_dups.items())[:5]:
#             print(f"  {token_bytes}: IDs = {ids}")
#     else:
#         print("  无重复")
    
#     print("\ntrain 中的重复 token:")
#     fast_dups = {k: v for k, v in fast_bytes_to_ids.items() if len(v) > 1}
#     if fast_dups:
#         for token_bytes, ids in list(fast_dups.items())[:5]:
#             print(f"  {token_bytes}: IDs = {ids}")
#     else:
#         print("  无重复")
    
#     # 比较词表大小
#     print(f"\n最终词表大小:")
#     print(f"  slow: {len(slow.itos)} tokens")
#     print(f"  fast: {len(fast.itos)} tokens")
#     print(f"  差异: {abs(len(slow.itos) - len(fast.itos))} tokens")

# def analyze_frequency_distribution(slow, fast, path):
#     """分析频数分布情况"""
#     print("\n" + "="*80)
#     print("初始 Pair 频数分布分析")
#     print("="*80)
    
#     # 获取初始的 pair counts
#     slow_stats = get_pair_stats_at_step(slow, path, 1)
#     fast_stats = get_pair_stats_at_step(fast, path, 1)
    
#     # 统计频数分布
#     slow_freq_dist = {}
#     fast_freq_dist = {}
    
#     for count in slow_stats.values():
#         slow_freq_dist[count] = slow_freq_dist.get(count, 0) + 1
    
#     for count in fast_stats.values():
#         fast_freq_dist[count] = fast_freq_dist.get(count, 0) + 1
    
#     # 显示前10个最高频数
#     print("\n频数分布 (显示前10个最高频数):")
#     print(f"{'Count':>10} {'slow pairs':>15} {'fast pairs':>15}")
#     print("-"*40)
    
#     all_counts = sorted(set(slow_freq_dist.keys()) | set(fast_freq_dist.keys()), reverse=True)[:10]
#     for count in all_counts:
#         slow_num = slow_freq_dist.get(count, 0)
#         fast_num = fast_freq_dist.get(count, 0)
#         print(f"{count:>10} {slow_num:>15} {fast_num:>15}")

# def main():
#     print("开始 BPE 训练对比测试...")
#     print(f"输入文件: {INPUT_PATH}")
    
#     # 运行两种实现
#     slow = run_slow(INPUT_PATH)
#     fast = run_fast(INPUT_PATH)
    
#     # 详细比较合并过程
#     diverge_point = compare_merges_detailed(slow, fast, INPUT_PATH)
    
#     # 检查 token 复用情况
#     check_token_reuse(slow, fast)
    
#     # 分析频数分布
#     analyze_frequency_distribution(slow, fast, INPUT_PATH)
    
#     # 如果需要更深入的调试，可以取消下面的注释
#     # if diverge_point:
#     #     print(f"\n建议：检查第 {diverge_point} 步的排序逻辑和频数统计")

# get_pair_stats_at_step
# debug_bpe.py
import pathlib, json, heapq, copy, re
from cs336_basics.tokenizer import BPETokenizer
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

INPUT_PATH   = FIXTURES_PATH / "corpus.en"
VOCAB_SIZE   = 500
SPECIALS     = ["<|endoftext|>"]

def run_slow(path):
    tk = BPETokenizer(vocab_size=VOCAB_SIZE, special_tokens=SPECIALS)
    tk.slow_train(path)
    return tk

def run_fast(path):
    tk = BPETokenizer(vocab_size=VOCAB_SIZE, special_tokens=SPECIALS)
    tk.train(path)
    return tk

def test_sorting_functions(tk):
    """测试 train 函数中的排序相关函数"""
    print("\n" + "="*80)
    print("测试排序函数")
    print("="*80)
    
    # 测试 bytes_desc 函数
    if hasattr(tk, 'bytes_desc'):
        print("\n测试 bytes_desc 函数:")
        test_bytes = [b' ', b' a', b'nd', b'd', b'abc']
        for b in test_bytes:
            desc = tk.bytes_desc(b)
            print(f"  bytes_desc({b}) = {desc}")
            print(f"    原始字节: {list(b)}")
            print(f"    反转字节: {list(desc)}")
    
    # 测试 pair_desc 函数
    if hasattr(tk, 'pair_desc'):
        print("\n测试 pair_desc 函数:")
        # 使用实际的 token IDs
        test_pairs = [
            (33, 101),    # (b' ', b'd')
            (258, 269),   # (b' a', b'nd')
        ]
        for pair in test_pairs:
            if pair[0] in tk.itos and pair[1] in tk.itos:
                a_bytes = tk.itos[pair[0]]
                b_bytes = tk.itos[pair[1]]
                desc = tk.pair_desc(pair)
                print(f"\n  pair_desc({pair}) = {desc}")
                print(f"    Token 1: {pair[0]} -> {a_bytes} -> {list(a_bytes)}")
                print(f"    Token 2: {pair[1]} -> {b_bytes} -> {list(b_bytes)}")

def analyze_heap_at_step(tk, path, step=32):
    """分析特定步骤时的堆状态"""
    print("\n" + "="*80)
    print(f"分析第 {step} 步的堆状态")
    print("="*80)
    
    # 重建到第 step 步之前的状态
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    if tk.special_tokens:
        special_pattern = f"({'|'.join(re.escape(s) for s in tk.special_tokens)})"
        text_parts = re.split(special_pattern, text)
    else:
        text_parts = [text]
    
    # 初始化 token groups
    initial_vocab_map = {v: k for k, v in tk.itos.items() if k < 256}
    token_groups = []
    for part in text_parts:
        if part in tk.special_tokens or not part:
            continue
        # 假设 pretokenize 返回字节序列的列表
        from cs336_basics.tokenizer import pretokenize
        words_in_bytes = pretokenize(part)
        for word in words_in_bytes:
            token_groups.append([initial_vocab_map[bytes([b])] for b in word])
    
    # 应用前 step-1 次合并
    for i in range(min(step-1, len(tk.merges))):
        p1_bytes, p2_bytes = tk.merges[i]
        p1_id = tk.stoi[p1_bytes]
        p2_id = tk.stoi[p2_bytes]
        new_id = tk.stoi[p1_bytes + p2_bytes]
        token_groups = tk._merge_pair_in_groups(token_groups, (p1_id, p2_id), new_id)
    
    # 获取当前的 pair counts
    pair_counts = tk._get_stats(token_groups)
    
    # 模拟 train 函数的堆初始化
    print("\n模拟 train 函数的堆初始化:")
    
    # 先定义内部函数（如果 tk 对象没有这些方法）
    def bytes_desc(b):
        return bytes(255 - x for x in b)
    
    def pair_desc(pair):
        a = tk.itos[pair[0]]
        b = tk.itos[pair[1]]
        max_len = max(len(a), len(b))
        a_pad = a + bytes([0] * (max_len - len(a)))
        b_pad = b + bytes([0] * (max_len - len(b)))
        return (bytes_desc(a_pad), bytes_desc(b_pad))
    
    # 构建堆
    heap = []
    for (a, b), cnt in pair_counts.items():
        heap_item = (
            -cnt,
            pair_desc((a, b)),
            a, b
        )
        heapq.heappush(heap, heap_item)
    
    # 找出频数为 609 的所有项
    target_count = 609
    print(f"\n频数为 {target_count} 的堆项:")
    matching_items = []
    temp_heap = heap.copy()
    
    while temp_heap:
        item = heapq.heappop(temp_heap)
        if -item[0] == target_count:
            matching_items.append(item)
            a, b = item[2], item[3]
            a_bytes = tk.itos[a]
            b_bytes = tk.itos[b]
            print(f"\n  堆项: count={-item[0]}, pair=({a}, {b})")
            print(f"    Bytes: ({a_bytes}, {b_bytes})")
            print(f"    pair_desc 结果: {item[1]}")
    
    # 比较这些项的排序
    print(f"\n按堆顺序排列（最小的会被选中）:")
    sorted_items = sorted(matching_items)
    for i, item in enumerate(sorted_items):
        a, b = item[2], item[3]
        a_bytes = tk.itos[a]
        b_bytes = tk.itos[b]
        marker = "← 会被选中" if i == 0 else ""
        print(f"  {i+1}. ({a_bytes}, {b_bytes}) {marker}")

def get_pair_stats_at_step(tk, path, step):
    """重现第 step 步时的 pair_counts 状态"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    if tk.special_tokens:
        special_pattern = f"({'|'.join(re.escape(s) for s in tk.special_tokens)})"
        text_parts = re.split(special_pattern, text)
    else:
        text_parts = [text]
    
    # 重建初始 token groups
    initial_vocab_map = {v: k for k, v in tk.itos.items() if k < 256}
    token_groups = []
    for part in text_parts:
        if part in tk.special_tokens or not part:
            continue
        from cs336_basics.tokenizer import pretokenize
        words_in_bytes = pretokenize(part)
        for word in words_in_bytes:
            token_groups.append([initial_vocab_map[bytes([b])] for b in word])
    
    # 应用前 step-1 次合并
    for i in range(min(step-1, len(tk.merges))):
        p1_bytes, p2_bytes = tk.merges[i]
        p1_id = tk.stoi[p1_bytes]
        p2_id = tk.stoi[p2_bytes]
        new_id = tk.stoi[p1_bytes + p2_bytes]
        token_groups = tk._merge_pair_in_groups(token_groups, (p1_id, p2_id), new_id)
    
    return tk._get_stats(token_groups)

def compare_sorting_logic(slow, fast):
    """直接比较两种排序逻辑"""
    print("\n" + "="*80)
    print("比较排序逻辑")
    print("="*80)
    
    # 创建测试数据
    test_pairs = [
        ((b' ', b'd'), 609),
        ((b' a', b'nd'), 609),
        ((b'e', b'd'), 609),
        ((b'a', b'b'), 609),
    ]
    
    print("\n测试数据:")
    for pair, count in test_pairs:
        print(f"  {pair}: count={count}")
    
    # slow_train 的排序
    print("\nslow_train 的排序逻辑:")
    print("  使用 max(key=lambda p: (count, p[0], p[1]))")
    slow_sorted = sorted(test_pairs, key=lambda x: (x[1], x[0][0], x[0][1]))
    print("  排序结果（升序）:")
    for i, (pair, count) in enumerate(slow_sorted):
        print(f"    {i+1}. {pair}")
    print(f"  max() 会选择: {slow_sorted[-1][0]}")
    
    # train 的排序（模拟）
    print("\ntrain 的排序逻辑（模拟）:")
    print("  使用 heapq（最小堆）+ pair_desc")
    
    def bytes_desc(b):
        return bytes(255 - x for x in b)
    
    def pair_desc_simple(pair):
        a, b = pair
        max_len = max(len(a), len(b))
        a_pad = a + bytes([0] * (max_len - len(a)))
        b_pad = b + bytes([0] * (max_len - len(b)))
        return (bytes_desc(a_pad), bytes_desc(b_pad))
    
    # 构建堆项
    heap_items = []
    for pair, count in test_pairs:
        item = (-count, pair_desc_simple(pair), pair)
        heap_items.append(item)
    
    # 排序（模拟堆的行为）
    heap_items.sort()
    print("  堆排序结果:")
    for i, item in enumerate(heap_items):
        print(f"    {i+1}. {item[2]}, pair_desc={item[1]}")
    print(f"  heappop() 会选择: {heap_items[0][2]}")

def compare_merges_detailed(slow, fast, path, max_show=10):
    """详细比较合并过程"""
    print("="*80)
    print(f"比较 slow_train 和 train 的合并过程")
    print(f"词表大小: {VOCAB_SIZE}, 特殊token: {SPECIALS}")
    print("="*80)
    
    print("\n初始词表检查:")
    print(f"slow 初始词表大小: {len([k for k in slow.itos if k < 256])}")
    print(f"fast 初始词表大小: {len([k for k in fast.itos if k < 256])}")
    
    for i in range(min(len(slow.merges), len(fast.merges))):
        slow_pair = slow.merges[i]
        fast_pair = fast.merges[i]
        
        if slow_pair != fast_pair:
            print(f"\n❌ 第 {i+1} 次 merge 开始分叉!")
            print("-"*80)
            
            # 获取这一步的 pair counts
            slow_stats = get_pair_stats_at_step(slow, path, i+1)
            fast_stats = get_pair_stats_at_step(fast, path, i+1)
            
            # 详细显示分叉点
            print(f"\n分叉详情:")
            print(f"{'':>10} {'Pair':>30} {'Count':>10} {'Token IDs':>20} {'New Token ID':>15}")
            print("-"*95)
            
            # slow 选择的 pair
            slow_p1_id = slow.stoi[slow_pair[0]]
            slow_p2_id = slow.stoi[slow_pair[1]]
            slow_new_id = slow.stoi[slow_pair[0] + slow_pair[1]]
            slow_count = slow_stats.get((slow_p1_id, slow_p2_id), 0)
            
            print(f"{'slow:':>10} {str(slow_pair):>30} {slow_count:>10} "
                  f"{f'({slow_p1_id}, {slow_p2_id})':>20} {slow_new_id:>15}")
            
            # fast 选择的 pair
            fast_p1_id = fast.stoi[fast_pair[0]]
            fast_p2_id = fast.stoi[fast_pair[1]]
            fast_new_id = fast.stoi[fast_pair[0] + fast_pair[1]]
            fast_count = fast_stats.get((fast_p1_id, fast_p2_id), 0)
            
            print(f"{'fast:':>10} {str(fast_pair):>30} {fast_count:>10} "
                  f"{f'({fast_p1_id}, {fast_p2_id})':>20} {fast_new_id:>15}")
            
            # 显示频数相同的其他候选
            print(f"\n频数为 {max(slow_count, fast_count)} 的所有候选 pairs:")
            candidates = []
            for (p1_id, p2_id), count in slow_stats.items():
                if count == max(slow_count, fast_count):
                    p1_bytes = slow.itos[p1_id]
                    p2_bytes = slow.itos[p2_id]
                    candidates.append(((p1_bytes, p2_bytes), count, (p1_id, p2_id)))
            
            # 按照两种排序规则排序
            slow_sorted = sorted(candidates, key=lambda x: (x[1], x[0][0], x[0][1]))
            
            print(f"\nslow_train 排序 (升序):")
            for j, (pair, count, ids) in enumerate(slow_sorted[-5:], 1):
                marker = "✓" if pair == slow_pair else " "
                print(f"  {marker} {j}. {str(pair):>30} count={count} ids={ids}")
            
            # 继续显示后续几步
            print(f"\n后续 {max_show} 步合并:")
            print(f"{'Step':>6} {'slow pair':>30} {'fast pair':>30} {'Match':>8}")
            print("-"*80)
            
            for j in range(i, min(i + max_show, min(len(slow.merges), len(fast.merges)))):
                s = slow.merges[j] if j < len(slow.merges) else None
                f = fast.merges[j] if j < len(fast.merges) else None
                match = "✓" if s == f else "✗"
                print(f"{j+1:>6} {str(s):>30} {str(f):>30} {match:>8}")
            
            return i + 1
    
    print("\n✅ 两个实现的合并序列完全一致!")
    return None

def main():
    print("开始 BPE 训练对比测试...")
    print(f"输入文件: {INPUT_PATH}")
    
    # 运行两种实现
    slow = run_slow(INPUT_PATH)
    fast = run_fast(INPUT_PATH)
    
    # 1. 测试排序函数
    test_sorting_functions(fast)
    
    # 2. 比较排序逻辑
    compare_sorting_logic(slow, fast)
    
    # 3. 详细比较合并过程
    diverge_point = compare_merges_detailed(slow, fast, INPUT_PATH, max_show=10)
    
    # 4. 如果发现分叉，分析该步骤的堆状态
    if diverge_point:
        analyze_heap_at_step(fast, INPUT_PATH, step=diverge_point)
    
    print("\n" + "="*80)
    print("调试完成")
    print("="*80)

if __name__ == "__main__":
    main()