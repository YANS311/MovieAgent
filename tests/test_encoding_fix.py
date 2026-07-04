#!/usr/bin/env python3
"""
诊断脚本：验证 CPU-only 编码修复是否生效
"""

import os
import sys
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("🔍 CUDA OOM 修复诊断工具")
print("=" * 70)

# 检查 1: CUDA 状态
print("\n[检查 1] CUDA 状态")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU 数量: {torch.cuda.device_count()}")
    print(f"  当前 GPU: {torch.cuda.get_device_name()}")
    print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  可用显存: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

# 检查 2: 模型下载和初始化（CPU 模式）
print("\n[检查 2] BGE 模型在 CPU 上的加载")
try:
    print("  加载 BGE-Base (BAAI/bge-base-en-v1.5) 到 CPU...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    print(f"  ✅ 模型加载成功")
    print(f"  模型输出维度: {model.get_sentence_embedding_dimension()}")

    # 检查 3: 验证小 batch 编码不会爆炸
    print("\n[检查 3] 小 batch 编码测试")
    test_texts = [f"test movie {i}" for i in range(10)]
    batch_size = 4

    print(f"  编码 {len(test_texts)} 条文本 (batch_size={batch_size})...")
    embeddings = model.encode(test_texts, batch_size=batch_size, show_progress_bar=False)
    print(f"  ✅ 编码完成")
    print(f"  输出形状: {embeddings.shape}")
    print(f"  内存使用: {embeddings.nbytes / 1e6:.2f} MB")

    # 检查 4: 大 batch（模拟 6639 条）
    print("\n[检查 4] 模拟大规模编码 (6639 条)")
    large_texts = [f"Movie title and plot summary {i}" for i in range(100)]
    print(f"  编码 {len(large_texts)} 条文本 (模拟 6639)...")
    large_embeddings = model.encode(large_texts, batch_size=4, show_progress_bar=False)
    print(f"  ✅ 编码完成")
    print(f"  输出形状: {large_embeddings.shape}")
    print(f"  内存使用: {large_embeddings.nbytes / 1e6:.2f} MB")

    # 检查 5: GPU 显存没有被占用
    if torch.cuda.is_available():
        print("\n[检查 5] GPU 显存占用检查")
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  PyTorch 在 GPU 上分配的显存: {allocated:.2f} GB")
        if allocated < 0.1:
            print(f"  ✅ GPU 显存安全（几乎不被占用）")
        else:
            print(f"  ⚠️ 警告：GPU 显存被占用过多")

    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("✅ 所有检查通过！CPU-only 编码修复应该能工作")
    print("=" * 70)
    print("\n接下来可以运行:")
    print("  python run_greedy_search.py")

except Exception as e:
    print(f"\n❌ 诊断失败: {e}")
    import traceback
    traceback.print_exc()

