# Cell-mem 测试指南

## 最快全量测试（22s）

```bash
python -m pytest tests/ -q --tb=short \
    --ignore=tests/test_stdio_clean.py \
    --ignore=tests/test_stdio_v2.py
```

226 个测试，22 秒内完成。

**为什么排除两个 test_stdio 文件？**
这两个是 MCP stdio 集成测试，需要启动真实子进程并等待握手，依赖网络和超时策略，不适合快速单元测试。

---

## 按模块分测

```bash
# 仅存储层（最快，~2s）
python -m pytest tests/test_sqlite_store.py tests/test_vector_store.py -q

# 仅记忆层（~5s）
python -m pytest tests/test_models.py tests/test_working_memory.py tests/test_embedding.py -q

# 仅 graph + 搜索（~3s）
python -m pytest tests/test_graph_store.py tests/test_search_engine.py tests/test_graph_stub.py -q

# 仅 consolidation（~10s，含 pattern detection）
python -m pytest tests/test_consolidation_phase2b.py tests/test_emotional_scorer_phase3.py -q

# 仅 Phase 3/4 功能（~5s）
python -m pytest tests/test_memory_system_phase2a.py tests/test_memory_system_phase3.py tests/test_memory_system_phase4.py -q

# 仅 reflection + replay + creative pool（~3s）
python -m pytest tests/test_reflection_engine.py tests/test_full_reflection.py tests/test_replay_engine.py tests/test_creative_pool.py -q

# 仅 procedural + LLM + falsifiable（~3s）
python -m pytest tests/test_procedural_memory.py tests/test_llm_client.py tests/test_falsifiable_conditions.py -q
```

---

## Mock 嵌入模型

所有 MemorySystem 测试都使用 `mock_embedding_model` fixture，返回确定性随机向量（384d），零 PyTorch 加载，零网络下载。

需要**真实嵌入**的测试场景？加 `@pytest.mark.real_embed` 标记，单独跑：
```bash
python -m pytest tests/ -m real_embed -v
```

---

## 首次设置

```bash
pip install -e ".[dev]"          # 安装测试依赖
```

---

## 性能基线

| 测试范围 | 耗时 | 测试数 |
|---------|------|--------|
| 全量（排除集成） | **22s** | 226 |
| 存储 + 记忆 | 7s | 60+ |
| Consolidation | 10s | 20+ |
| Phase 3/4 | 5s | 30+ |
