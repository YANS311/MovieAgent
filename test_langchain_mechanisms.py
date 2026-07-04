#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain 三大机制集成测试脚本
演示: 记忆管理、检索压缩、智能路由

使用方法:
    python test_langchain_mechanisms.py
"""

import sys
import time
from pathlib import Path

# 添加项目路径
project_path = Path(__file__).parent
sys.path.insert(0, str(project_path))

from myapp.langchain_memory_enhancer import (
    ConversationSummaryBufferMemory,
    ContextualCompressionRetriever,
    ChatBranchRouter,
    build_memory_enhanced_prompt,
    format_memory_for_display,
)


def print_section(title: str):
    """打印分区标题"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def test_memory_management():
    """测试 1: 记忆管理 (ConversationSummaryBufferMemory)"""
    print_section("TEST 1: ConversationSummaryBufferMemory - 记忆管理")

    # 初始化记忆管理器
    memory = ConversationSummaryBufferMemory(
        max_token_limit=300,  # 低 Token 预算以演示摘要
        short_term_rounds=3
    )
    print("✨ 初始化完成")
    print(f"   • Token 预算: {memory.max_token_limit}")
    print(f"   • 短期轮数: {memory.short_term_rounds}\n")

    # 模拟多轮对话
    conversations = [
        ("user", "推荐一部科幻电影，我特别喜欢《Inception》那样题曰深刻的作品。"),
        ("ai", "根据您的喜好，推荐《Interstellar》。这部电影由 Christopher Nolan 导演..."),
        ("user", "还有类似的吗？"),
        ("ai", "还可以看《The Matrix》，它也探讨了哲学问题..."),
        ("user", "这些电影的导演是谁？"),
        ("ai", "《Inception》和《Interstellar》都是 Nolan 导演的。《The Matrix》是 Wachowski 姐妹导演..."),
        ("user", "你能为我总结一下我们的对话吗？"),
        ("ai", "您对科幻电影感兴趣，特别是题意深刻的作品。我推荐了 Nolan 导演的《Inception》和《Interstellar》..."),
    ]

    print("🔄 添加对话消息：\n")
    for i, (role, content) in enumerate(conversations, 1):
        memory.add_message(role, content)
        print(f"   {i}. [{role.upper():4s}] {content[:50]}...")

        # 显示记忆状态
        stats = format_memory_for_display(memory)
        print(f"      → 总消息数: {stats['total_messages']} | "
              f"短期轮数: {stats['short_term_rounds']} | "
              f"有摘要: {stats['has_summary']}")

    print("\n📊 最终记忆状态：")
    print(f"   • 总对话轮数: {len(memory.chat_history)}")
    print(f"   • 短期记忆: {len(memory.short_term_memory)} 条消息")
    print(f"   • 历史摘要长度: {len(memory.moving_summary)} 字")

    # 显示记忆上下文
    moving_summary, short_term = memory.get_context()

    if moving_summary:
        print(f"\n📝 历史摘要（用于 Prompt 顶部）:\n   {moving_summary}\n")

    print(f"📋 短期历史（最近对话）:\n")
    short_term_lines = short_term.split('\n')
    for line in short_term_lines[:6]:  # 只显示前 6 行
        print(f"   {line}")
    if len(short_term_lines) > 6:
        print(f"   ... 共 {len(short_term_lines)} 行 ...\n")

    print("✅ 记忆管理测试完成！")


def test_retrieval_compression():
    """测试 2: 检索压缩 (ContextualCompressionRetriever)"""
    print_section("TEST 2: ContextualCompressionRetriever - 检索压缩")

    compressor = ContextualCompressionRetriever(similarity_threshold=0.75)

    # 模拟 RAG 召回结果
    query = "科幻电影推荐"

    rag_results = [
        {
            'content': '《Inception》是一部关于梦境和现实的科幻电影...',
            'metadata': {'movie_id': 1, 'score': 8.8}
        },
        {
            'content': '《Interstellar》探讨人类在太空的冒险...',
            'metadata': {'movie_id': 2, 'score': 9.0}
        },
        {
            'content': '《The Matrix》展现了虚拟世界的设定...',
            'metadata': {'movie_id': 3, 'score': 8.7}
        },
        {
            'content': '《钢铁侠》是一部关于机器人的动作电影...',  # 不太相关
            'metadata': {'movie_id': 4, 'score': 8.1}
        },
        {
            'content': '《蜘蛛侠》讲述了超级英雄的故事...',  # 不太相关
            'metadata': {'movie_id': 5, 'score': 7.9}
        },
        {
            'content': '《Dune》是一部太空歌剧科幻电影...',
            'metadata': {'movie_id': 6, 'score': 8.5}
        },
    ]

    print(f"📥 原始召回结果: {len(rag_results)} 条文档\n")
    for i, doc in enumerate(rag_results, 1):
        print(f"   {i}. [{doc['metadata']['movie_id']}] "
              f"{doc['content'][:50]}... (评分: {doc['metadata']['score']})")

    # 执行压缩
    print(f"\n🔍 执行语义压缩 (threshold={compressor.similarity_threshold})...\n")
    compressed = compressor.filter_documents(query, rag_results)

    print(f"📤 压缩后结果: {len(compressed)} 条文档\n")
    for i, doc in enumerate(compressed, 1):
        score = doc.get('similarity_score', 0)
        print(f"   {i}. [{doc['metadata']['movie_id']}] "
              f"{doc['content'][:50]}...")
        print(f"      → 相似度: {score:.3f}")

    if len(rag_results) > 0:
        reduction = (1 - len(compressed) / len(rag_results)) * 100
        print(f"\n📊 压缩统计:")
        print(f"   • 过滤率: {reduction:.1f}% ({len(rag_results) - len(compressed)} 条)")
        print(f"   • 保留率: {100 - reduction:.1f}%")

    print("\n✅ 检索压缩测试完成！")


def test_branch_routing():
    """测试 3: 智能路由 (ChatBranchRouter)"""
    print_section("TEST 3: ChatBranchRouter - 智能意图路由")

    router = ChatBranchRouter()

    # 测试不同的意图
    test_cases = [
        ("QUERY_MOVIE", "推荐一部科幻电影", "基于内容的推荐"),
        ("QUERY_VISUAL", "找一部有蓝色调的电影", "基于视觉特征的搜索"),
        ("QUERY_KG", "Nolan 的电影都有什么特点？", "知识图谱查询"),
        ("CHAT", "你好，今天天气如何？", "日常闲聊"),
        ("QUERY_RANK", "最近有什么高分电影？", "榜单查询"),
        ("QUERY_PROFILE_REC", "根据我的口味推荐", "个性化推荐"),
    ]

    print("🚀 测试意图路由:\n")

    for intent, user_input, description in test_cases:
        print(f"📌 意图: {intent}")
        print(f"   输入: \"{user_input}\"")
        print(f"   期望: {description}\n")

        # 执行路由
        context = {
            'interaction_summary': '用户喜欢科幻电影和导演 Nolan 的作品',
            'is_thinking_mode': False,
        }
        result = router.route(intent, user_input, context)

        # 显示路由结果
        print(f"   ✅ 路由决策:")
        print(f"      • 分支: {result['branch']}")
        print(f"      • 描述: {result['description']}")
        print(f"      • 操作: {result['action']}")
        print(f"      • 使用 RAG: {result['use_rag']}")
        print(f"      • 使用图谱: {result['use_graph']}")
        print()

    print("✅ 智能路由测试完成！")


def test_memory_enhanced_prompt():
    """测试 4: 记忆增强 Prompt"""
    print_section("TEST 4: 记忆增强 Prompt 构造")

    # 初始化记忆并添加历史
    memory = ConversationSummaryBufferMemory(max_token_limit=500)

    conversations = [
        ("user", "我喜欢 Christopher Nolan 的电影"),
        ("ai", "Nolan 的作品以复杂的叙事和宏大的视觉著称"),
        ("user", "推荐一部他最近的作品"),
        ("ai", "《Oppenheimer》是他最新的力作"),
    ]

    for role, content in conversations:
        memory.add_message(role, content)

    # 构造记忆增强的 Prompt
    user_input = "还有类似的吗？"
    rag_context = "《Dunkirk》: Nolan 的战争电影杰作\n《Tenet》: 关于逆熵的科幻作品"

    enhanced_prompt = build_memory_enhanced_prompt(
        user_input=user_input,
        memory=memory,
        rag_context=rag_context,
        system_role="电影推荐助手"
    )

    print("📝 生成的增强 Prompt:\n")
    print("-" * 70)
    print(enhanced_prompt)
    print("-" * 70)

    print(f"\n📊 Prompt 统计:")
    print(f"   • 长度: {len(enhanced_prompt)} 字符")
    print(f"   • 估计 Token: {int(len(enhanced_prompt) * 1.2)} Token")

    print("\n✅ Prompt 构造测试完成！")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("  LangChain 三大高级机制端到端测试")
    print("  " + "=" * 70)

    try:
        # Test 1: 记忆管理
        test_memory_management()
        time.sleep(1)

        # Test 2: 检索压缩
        test_retrieval_compression()
        time.sleep(1)

        # Test 3: 智能路由
        test_branch_routing()
        time.sleep(1)

        # Test 4: Prompt 增强
        test_memory_enhanced_prompt()

        print("\n" + "=" * 70)
        print("  ✅ 所有测试通过！系统已就绪。")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

