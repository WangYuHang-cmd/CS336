# cs336_basics/debug_tokenizer.py
import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.tokenizer import BPETokenizer
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH, FIXTURES_PATH
import tiktoken

def debug_token_differences():
    """调试token 628 vs 198的问题"""
    print("=== 调试Token差异 ===")
    
    # 加载tokenizer
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, 
        merges_path=MERGES_PATH, 
        special_tokens=["<|endoftext|>"]
    )
    
    print(MERGES_PATH)
    
    for id,item in enumerate(tokenizer.merges):
        if b'\n' in item:
            print(id,end=": ")
            print(item)

    reference_tokenizer = tiktoken.get_encoding("gpt2")
    
    print(f"My tokenizer \\n\\n: {tokenizer.stoi[b'\n\n']}")
    
    # 1. 检查特定token的内容
    print(f"\nToken 198: {repr(tokenizer.itos.get(198, 'NOT FOUND'))}")
    print(f"Token 628: {repr(tokenizer.itos.get(628, 'NOT FOUND'))}")
    
    # 2. 测试各种换行符组合
    test_cases = [
        "\n",
        "\n\n",
        "\n\n\n",
        "a\n\nb",
        "<|endoftext|>\n\n",
        "\n\n<|endoftext|>",
    ]
    
    print("\n=== 测试换行符编码 ===")
    for test in test_cases:
        our_ids = tokenizer.encode(test)
        ref_ids = reference_tokenizer.encode(test, allowed_special={"<|endoftext|>"})
        match = "✓" if our_ids == ref_ids else "✗"
        print(f"{match} Input: {repr(test)}")
        print(f"  Our encoding: {our_ids}")
        print(f"  Ref encoding: {ref_ids}")
        if our_ids != ref_ids:
            print(f"  Our tokens: {[repr(tokenizer.itos[id]) for id in our_ids]}")
            print(f"  Ref tokens: {[repr(reference_tokenizer.decode([id])) for id in ref_ids]}")
    
    # 3. 检查相关的merge规则
    print("\n=== 相关Merge规则 ===")
    newline_byte = b'\n'
    for i, (left, right) in enumerate(tokenizer.merges[:100]):
        if newline_byte in left or newline_byte in right:
            merged = left + right
            token_id = tokenizer.stoi.get(merged, "NOT IN VOCAB")
            print(f"Merge {i}: {repr(left)} + {repr(right)} = {repr(merged)} (ID: {token_id})")

def debug_specific_file():
    """调试特定失败的测试文件"""
    print("\n=== 调试特定文件 ===")
    
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, 
        merges_path=MERGES_PATH, 
        special_tokens=["<|endoftext|>"]
    )
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    
    # 检查 special_token_double_newlines_non_whitespace.txt
    corpus_path = FIXTURES_PATH / "special_token_double_newlines_non_whitespace.txt"
    with open(corpus_path, 'rb') as f:
        content_bytes = f.read()
    
    print(f"\n文件内容 (bytes): {repr(content_bytes)}")
    content = content_bytes.decode('utf-8')
    print(f"文件内容 (string): {repr(content)}")
    
    our_ids = tokenizer.encode(content)
    ref_ids = reference_tokenizer.encode(content, allowed_special={"<|endoftext|>"})
    
    print(f"\nOur encoding: {our_ids}")
    print(f"Ref encoding: {ref_ids}")
    
    # 找出差异位置
    for i, (our_id, ref_id) in enumerate(zip(our_ids, ref_ids)):
        if our_id != ref_id:
            print(f"\n差异在位置 {i}:")
            print(f"  Our token {our_id}: {repr(tokenizer.itos[our_id])}")
            print(f"  Ref token {ref_id}: {repr(reference_tokenizer.decode([ref_id]))}")
            break

def debug_merge_process():
    """调试BPE合并过程"""
    print("\n=== 调试BPE合并过程 ===")
    
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, 
        merges_path=MERGES_PATH, 
        special_tokens=["<|endoftext|>"]
    )
    
    # 测试一个简单的双换行符
    test_bytes = b'\n\n'
    
    # 模拟_encode_ordinary_text的过程
    tokens = [bytes([b]) for b in test_bytes]
    print(f"初始tokens: {[repr(t) for t in tokens]}")
    
    # 逐步应用merge规则
    merge_found = False
    for i, (left, right) in enumerate(tokenizer.merges[:50]):
        original_tokens = tokens.copy()
        j = 0
        changed = False
        while j < len(tokens) - 1:
            if tokens[j] == left and tokens[j + 1] == right:
                tokens[j] = left + right
                tokens.pop(j + 1)
                changed = True
            else:
                j += 1
        
        if changed:
            print(f"\nMerge规则 {i}: {repr(left)} + {repr(right)} = {repr(left + right)}")
            print(f"  应用前: {[repr(t) for t in original_tokens]}")
            print(f"  应用后: {[repr(t) for t in tokens]}")
            if left == b'\n' and right == b'\n':
                merge_found = True
                print("  *** 找到双换行符合并规则! ***")
    
    if not merge_found:
        print("\n*** 警告：没有找到合并两个换行符的规则! ***")
    
    # 检查最终的token ID
    print(f"\n最终tokens: {[repr(t) for t in tokens]}")
    final_ids = []
    for token in tokens:
        if token in tokenizer.stoi:
            token_id = tokenizer.stoi[token]
            final_ids.append(token_id)
            print(f"Token {repr(token)} -> ID {token_id}")
        else:
            print(f"Token {repr(token)} NOT IN VOCABULARY!")
    
    print(f"\n最终IDs: {final_ids}")

def debug_encode_iterable_memory():
    """调试encode_iterable的内存问题"""
    print("\n=== 调试encode_iterable ===")
    
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, 
        merges_path=MERGES_PATH, 
        special_tokens=[]
    )
    
    # 创建一个小的测试迭代器
    test_lines = ["Hello world\n", "This is a test\n", "With multiple lines\n"]
    
    print("测试encode_iterable:")
    all_ids = []
    for i, token_id in enumerate(tokenizer.encode_iterable(test_lines)):
        all_ids.append(token_id)
        if i < 20:  # 只打印前20个
            print(f"  Yielded token {i}: {token_id}")
    
    # 对比直接编码
    direct_text = "".join(test_lines)
    direct_ids = tokenizer.encode(direct_text)
    
    print(f"\nDirect encoding length: {len(direct_ids)}")
    print(f"Iterable encoding length: {len(all_ids)}")
    print(f"Match: {all_ids == direct_ids}")

def check_vocab_loading():
    """检查vocab是否正确加载"""
    print("\n=== 检查Vocab加载 ===")
    
    # 直接检查GPT-2 vocab文件
    with open(VOCAB_PATH) as f:
        gpt2_vocab = json.load(f)
    
    # 查找包含换行符的tokens
    print("\nGPT-2 vocab中ID 198和628的内容:")
    if '198' in [str(v) for k, v in gpt2_vocab.items()]:
        for k, v in gpt2_vocab.items():
            if v == 198:
                print(f"  Token 198 in vocab: {repr(k)}")
            if v == 628:
                print(f"  Token 628 in vocab: {repr(k)}")

if __name__ == "__main__":
    debug_token_differences()
    print('=================================================\n')
    debug_specific_file()
    # debug_merge_process()
    # debug_encode_iterable_memory()
    # check_vocab_loading()


'''
import tiktoken

# 1. 加载官方的、内建的 gpt2 分词器
# 这不读取任何外部文件，使用的是tiktoken库编译好的版本
try:
    tokenizer = tiktoken.get_encoding("gpt2")
    print("成功加载 tiktoken 内建的 'gpt2' 分词器。\n")
except Exception as e:
    print(f"加载分词器失败: {e}")
    exit()

# 2. 测试一：对 '\n\n' 字符串进行编码
# 这是为了查看它的实际行为
test_string = "\n\n"
encoded_ids = tokenizer.encode(test_string)

print(f"--- 测试一：编码字符串 '\\n\\n' ---")
print(f"输入: {repr(test_string)}")
print(f"输出的Token IDs: {encoded_ids}")
print("-" * 30)

# 3. 测试二：直接检查 b'\n\n' 这个合并后的token是否存在于词汇表中
# 我们尝试将 b'\n\n' 作为一个“单一token”进行编码
# 如果它存在于词汇表中，这个调用会成功并返回一个ID
# 如果它不存在，这个调用会失败并抛出 KeyError
merged_bytes = b'\n\n'

print(f"--- 测试二：检查 token {repr(merged_bytes)} 是否存在 ---")
try:
    # encode_single_token 是最严格的检查方式
    single_token_id = tokenizer.encode_single_token(merged_bytes)
    print(f"成功！词汇表中存在 {repr(merged_bytes)}，其ID为: {single_token_id}")
    print("这与预期的标准 gpt2 行为不符。")
except KeyError:
    print(f"失败！正如预期，词汇表中不存在 {repr(merged_bytes)} 这个单一token。")
    print("tiktoken 抛出了 KeyError，证明它无法将 b'\\n\\n' 视为一个整体。")
except Exception as e:
    print(f"发生了意外的错误: {e}")

print("-" * 30)
'''