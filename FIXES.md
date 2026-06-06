# Cell-mem 修复记录

## 第 5 轮 — 2026-06-06 — Codex v3 测试报告

| # | 问题 | 修复 | 文件 |
|---|------|------|------|
| 1 | associate 不校验 ID 存在，幽灵节点 | 前置检查：遍历 4 层确认两 ID 均存在 | `memory_system.py` |
| 2 | working_memory 表缺 tags_json 列 | DDL 加列 + 迁移 + working.py add() 写入 | `sqlite_store.py` `working.py` |
| 3 | ensure_loaded() 线程不安全，双线程同时加载 | 加 `threading.Lock` 保护 | `embedding/local.py` |

---

## 第 4 轮 — 2026-06-06 — Codex v2.0 功能测试

| # | 问题 | 修复 | 文件 |
|---|------|------|------|
| 1 | ONNX 失败后 `self._model` 未清空，PyTorch 回退时调错方法 | `_try_onnx()` 成功后才赋值 | `embedding/local.py` |
| 2 | recall 无向量时崩溃报错 | 嵌入不可用时降级纯 FTS5 | `memory_system.py` |
| 3 | EmbeddingWorker 不自动启动 | `__init__()` 内自启，去掉 server.py 重复调用 | `memory_system.py` `server.py` |
| 4 | 向量空结果 + pending 时不降级 | 检测 pending>0 自动切 FTS5 | `memory_system.py` |
| 5 | LLM api_key 无配置入口 | 支持 `CELL_MEM_LLM_API_KEY` → `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | `memory_system.py` |
| 6 | DB 路径不支持环境变量 | `--db` 默认读 `$CELL_MEM_DB` | `server.py` |
| 7 | FTS5 召回结果重复 4-5 条 | `recall_by_keyword` + `recall` 两层按 ID 去重 | `episodic.py` `memory_system.py` |
| 8 | associate 错误返回空字符串 | `str(exc)` 为空时用 `type(exc).__name__` 兜底 | `memory_system.py` |
| 9 | 向量搜索 mock 噪声匹配不相关内容 | FTS5 关键词后过滤：向量结果 ∩ FTS5 命中集 | `memory_system.py` |

---

## 第 3 轮 — 2026-06-05 — 用户实际使用报告

| # | 问题 | 修复 | 文件 |
|---|------|------|------|
| 1 | `memory_save` tags 参数丢失 | options 中提取 tags → 写入 tags_json | `memory_system.py` |
| 2 | SentenceTransformer 加载时 stderr 洪流 | suppress 5 个 logger + stdout/stderr redirect | `embedding/local.py` |
| 3 | sqlite-vec Errno 22 诊断不足 | 增加安装指导 + 版本检测 | `sqlite_store.py` |
| 4 | ChromaDB 空 metadata 被拒 | `{"_id": id}` placeholder 兜底 | `vector_store.py` |
| 5 | `memory_status` 不显示 DB 路径 | 新增 `db_path` 字段 | `memory_system.py` |
| 6 | stdio 模式预加载模型阻塞握手 | `if args.preload:` 替代 `if args.preload or not args.http:` | `server.py` |
| 7 | `semantic.update()` 残留同步嵌入 | 内容变更时 embedding=NULL，worker 异步补 | `semantic.py` |

---

## 第 2 轮 — 2026-06-05 — Codex Phase 2 审查

| # | 问题 | 修复 | 文件 |
|---|------|------|------|
| 1 | 调度器状态不持久化 | ✅ 误报 — `_restore_state()` 早已存在 | `scheduler.py` |

---

## 第 1 轮 — 2026-06-04 — Codex Phase 3+4 审查 + Aether 全面审查

| # | 问题 | 来源 | 文件 |
|---|------|------|------|
| 1 | `process_results` dict 嵌套错误 | Aether | `reflection/engine.py` |
| 2 | `match_by_keyword` FTS5 隐式 AND | Aether | `procedural.py` |
| 3 | `_sanitize_for_prompt` 缺失 | Codex | `replay/engine.py` |
| 4 | `record_feedback` 操作顺序不当 | Codex | `creative_pool.py` |
| 5 | 乐观锁缺失 | Codex | `creative_pool.py` |
| 6 | SSRF 防护缺失 | Codex | `llm/backends.py` |
| 7 | LLM 输出未校验 | Codex | `llm/backends.py` |
| 8 | 环境键值无上限 | Codex | `conditions/evaluator.py` |
| 9 | `--api-key` CLI 参数缺失 | Codex | `server.py` |
| 10 | `explore_ratio` 未传递 | Codex | `memory_system.py` |
| 11 | HTTP 模式无认证警告 | Codex | `server.py` |
| 12 | 异常处理日志级别低 | Codex | `scheduler.py` `replay/engine.py` |
| 13 | 创意池无 max_size | Codex | `creative_pool.py` |
