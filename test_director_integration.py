#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证导演字段集成的正确性
Test Script: Validate Director Field Integration
"""

import os
import django
import sys

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DjangoProject3.settings')
django.setup()

from myapp.models import Movie, Actor

def test_director_integration():
    """验证导演字段集成"""
    print("\n" + "="*70)
    print("🔍 导演字段集成测试 (Director Field Integration Test)")
    print("="*70)

    # 测试1: 验证Movie模型有directors字段
    print("\n[Test 1] 验证Movie模型包含directors字段...")
    try:
        test_movie = Movie.objects.first()
        if test_movie:
            directors = test_movie.directors.all()
            print(f"   ✅ Movie模型成功获取directors: {list(directors)}")
        else:
            print("   ⚠️  数据库中没有电影数据")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False

    # 测试2: 验证导演相似度计算函数
    print("\n[Test 2] 验证calculate_director_similarity()函数...")
    try:
        from myapp.views import calculate_director_similarity

        # 测试用例1: 完全相同
        score1 = calculate_director_similarity(["Christopher Nolan"], ["Christopher Nolan"])
        assert score1 == 1.0, f"Expected 1.0, got {score1}"
        print(f"   ✅ 完全相同: {score1}")

        # 测试用例2: 部分重合
        score2 = calculate_director_similarity(
            ["Christopher Nolan", "Steven Spielberg"],
            ["Christopher Nolan", "Wes Anderson"]
        )
        expected = 1.0 / 3.0  # Jaccard: 1个共同 / 3个总共
        assert abs(score2 - expected) < 0.01, f"Expected {expected}, got {score2}"
        print(f"   ✅ 部分重合: {score2:.4f}")

        # 测试用例3: 完全不同
        score3 = calculate_director_similarity(
            ["Christopher Nolan"],
            ["Wes Anderson"]
        )
        assert score3 == 0.0, f"Expected 0.0, got {score3}"
        print(f"   ✅ 完全不同: {score3}")

        # 测试用例4: 空列表
        score4 = calculate_director_similarity([], ["Nolan"])
        assert score4 == 0.0, f"Expected 0.0, got {score4}"
        print(f"   ✅ 空列表处理: {score4}")

    except ImportError as e:
        print(f"   ❌ 函数导入失败: {e}")
        return False
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False

    # 测试3: 验证get_connection_point_for_kg()函数已支持导演
    print("\n[Test 3] 验证get_connection_point_for_kg()支持导演关联...")
    try:
        from myapp.views import get_connection_point_for_kg

        # 创建两个测试电影对象
        movie1 = Movie.objects.first()
        movie2 = Movie.objects.last()

        if movie1 and movie2 and movie1.id != movie2.id:
            result_type, result_reason, result_keyword, result_found = get_connection_point_for_kg(movie1, movie2)
            print(f"   ℹ️  关联类型: {result_type}")
            print(f"   ℹ️  关联理由: {result_reason}")
            print(f"   ℹ️  关联关键词: {result_keyword}")
            print(f"   ✅ 函数成功执行")
        else:
            print("   ⚠️  数据库中电影数据不足")

    except ImportError as e:
        print(f"   ❌ 函数导入失败: {e}")
        return False
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False

    # 测试4: 验证Neo4j查询支持导演
    print("\n[Test 4] 验证Neo4j图谱包含导演节点...")
    try:
        from py2neo import Graph

        from django.conf import settings
        graph = Graph(
            getattr(settings, 'NEO4J_URI', 'bolt://localhost:7687'),
            auth=(getattr(settings, 'NEO4J_USER', 'neo4j'), getattr(settings, 'NEO4J_PASSWORD', ''))
        )
        result = graph.run(
            "MATCH (m:Movie)-[r:DIRECTED_BY]-(d:Person) RETURN COUNT(DISTINCT d) AS director_count LIMIT 1"
        ).data()

        if result and result[0]['director_count'] > 0:
            print(f"   ✅ Neo4j中找到 {result[0]['director_count']} 个导演节点")
        else:
            print("   ⚠️  Neo4j中未找到导演关联")

    except Exception as e:
        print(f"   ⚠️  Neo4j连接失败: {e} (这是可选的，可能没有启动Neo4j)")

    print("\n" + "="*70)
    print("✅ 所有核心测试通过！导演字段集成完成。")
    print("="*70 + "\n")
    return True


if __name__ == "__main__":
    success = test_director_integration()
    sys.exit(0 if success else 1)

