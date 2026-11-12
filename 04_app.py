"""
Step 4: RAG Web 应用（Gradio 美化版）
功能：
  - 多知识库：docs/ 下每个子目录一个库，右上角切换，各自独立检索
  - 聊天问答（流式输出 + 参考片段可视化）
  - 上传文档到当前知识库
  - 一键清空旧索引 + 重建全部知识库
  - 知识库状态 / 文档清单（按库分组）
运行：.venv\\Scripts\\python.exe 04_app.py
浏览器打开：http://127.0.0.1:7860
"""
import os
import html
import time
import shutil
import importlib.util

import gradio as gr

import rag_core as core

# —— 全局配置：切片 / 检索参数统一由 rag_core 管理，01/02/03 读的是同一份 ——
DOCS_DIR = core.DOCS_DIR
DB_DIR = core.DB_DIR
CONFIG_PATH = core.CONFIG_PATH

CFG = core.load_settings()
CHUNK_SIZE = CFG["chunk_size"]
CHUNK_OVERLAP = CFG["chunk_overlap"]
ADAPTIVE = CFG["adaptive"]
RERANK = CFG["rerank"]
TOP_K = CFG["top_k"]

# 界面上没有暴露的检索参数：启动时快照一次，之后显式传给 _cfg_from_ui，
# 不再在每次交互时重读磁盘（避免「读回的旧值覆盖界面新值」）
SEARCH_PARAMS = {k: CFG[k] for k in ("top_k", "candidate_k", "rrf_k",
                                     "bm25_weight", "vector_weight", "rerank_model")}

# 知识库列表（docs/ 下的子目录），按文档总量从大到小排；
# 默认库用 core.default_corpus()，与 resolve_corpus(None) 保持同一个口径
_CORPORA_DIRS = core.discover_corpora()
CORPORA = sorted(_CORPORA_DIRS, key=lambda n: -core.corpus_size(_CORPORA_DIRS[n]))
ACTIVE_CORPUS = core.default_corpus(_CORPORA_DIRS)

RERANK_CHOICES = [
    ("不精排（RRF 粗排）", "off"),
    ("向量重排（bge-m3 余弦）", "embed"),
    ("LLM 精排（qwen2.5:7b，较慢）", "llm"),
    ("bge-reranker 交叉编码器（需下载模型）", "cross"),
]

llm = core.get_llm()
PROMPT = core.build_prompt()

EMPTY_SOURCES = """
<div class="src-empty">
  <div class="src-empty-icon">🔎</div>
  <div>提问后，这里会显示本次检索命中的原始片段，方便核对答案来源</div>
</div>
"""

WELCOME = [{
    "role": "assistant",
    "content": "你好，我是你的文档助手 📄\n\n我已经读过 docs/ 里的资料，"
               "可以直接问我制度相关的问题。答案只来自你的知识库，"
               "查不到我会明说「无法回答」，不会编造。\n\n"
               "不知道从哪开始？点下面的示例问题试试。",
}]

# —— 全局缓存：每个知识库一个检索器 + 一个 Chroma 实例（避免反复建连接占用文件句柄） ——
_RETRIEVERS = {}
_VS = {}
# 建检索器时的索引签名：别的进程（01/02/03）重建索引后，用它判断缓存已过期
_RETRIEVER_SIG = {}


# ------------------------- 数据层（全部委托 rag_core，参数只有一个来源） -------------------------
def _load_docs(corpus=None):
    _, path, recursive = core.resolve_corpus(corpus)
    return core.load_docs(path, recursive=recursive)


def _load_chunks(corpus=None):
    """加载 + 分块，参数取自配置文件，与 01/02/03 完全一致"""
    n_docs, chunks = core.load_chunks(core.load_settings(), corpus=corpus)
    return n_docs, chunks


def _drop_cache():
    """丢掉所有检索器 / 向量句柄 / 签名（重建或清空索引后必须调）"""
    _RETRIEVERS.clear()
    _VS.clear()
    _RETRIEVER_SIG.clear()


def init_retriever(corpus=None, cfg=None):
    """初始化（或刷新）指定知识库的检索器：复用 Chroma 实例 + 重建 BM25

    集合为空时直接标记为 None —— 清空索引后如果不判断，会构造出一个
    「空检索器」，提问时静默返回 0 条片段，看起来像「文档里没有答案」，
    而不是「你还没建索引」。
    """
    cfg = core.normalize(cfg or core.load_settings())
    name, _, _ = core.resolve_corpus(corpus)
    if name not in _VS:
        _VS[name] = core.open_vectorstore(cfg, corpus=name)
    _RETRIEVER_SIG[name] = core.index_signature(name)
    try:
        if _VS[name]._collection.count() == 0:
            _RETRIEVERS[name] = None
            return False
    except Exception:
        pass
    _RETRIEVERS[name] = core.build_retriever(cfg, vs=_VS[name], corpus=name)
    return _RETRIEVERS[name] is not None


def get_retriever(corpus=None):
    """按需懒加载；索引被别的进程改过时自动丢弃缓存重建。"""
    name, _, _ = core.resolve_corpus(corpus)
    sig = core.index_signature(name)
    if name in _RETRIEVERS and _RETRIEVER_SIG.get(name) != sig:
        # 内存里的向量句柄和 BM25 都指向旧索引，继续用会返回过期片段
        print(f"[app] 检测到「{name}」索引已变更，重建检索器")
        _RETRIEVERS.pop(name, None)
        _VS.pop(name, None)
    if name not in _RETRIEVERS:
        try:
            init_retriever(name)
        except Exception as e:
            print(f"[app] 知识库 {name} 初始化失败：{e}")
            _RETRIEVERS[name] = None
            _RETRIEVER_SIG[name] = sig
    return _RETRIEVERS.get(name)


# ------------------------- 视图层（HTML 片段） -------------------------
def _status_html(kind, title, meta):
    icon = {"ok": "✓", "warn": "!", "error": "×"}.get(kind, "•")
    return f"""
    <div class="status status-{kind}">
      <div class="status-left">
        <span class="status-dot">{icon}</span>
        <div>
          <div class="status-title">{title}</div>
          <div class="status-meta">{meta}</div>
        </div>
      </div>
    </div>
    """


def _docs_html(corpus=None):
    """按知识库分组展示文档"""
    groups = []
    corpora = core.discover_corpora()
    for name, path in corpora.items():
        files = []
        for fname in sorted(os.listdir(path)):
            full = os.path.join(path, fname)
            if not (os.path.isfile(full) and fname.lower().endswith(core.DOC_EXTS)):
                continue
            size = os.path.getsize(full) / 1024
            ext = fname.rsplit(".", 1)[-1].lower()
            icon = {"pdf": "📕", "md": "📘", "txt": "📄"}.get(ext, "📎")
            active = "" if (corpus is None or name == corpus) else ' style="opacity:.55"'
            files.append(
                f'<div class="doc-item"{active}><span class="doc-icon">{icon}</span>'
                f'<span class="doc-name" title="{html.escape(fname)}">{html.escape(fname)}</span>'
                f'<span class="doc-size">{size:.1f} KB</span></div>'
            )
        if files:
            groups.append(
                f'<div class="doc-group">{html.escape(name)} · {len(files)}</div>'
                f'<div class="doc-list">{"".join(files)}</div>'
            )
    if not groups:
        return '<div class="doc-empty">docs/ 里还没有文档，先上传一份吧</div>'
    return "".join(groups)


def get_status(corpus=None):
    """返回 (状态卡片 HTML, 文档清单 HTML)；corpus 指定当前使用的知识库"""
    cfg = core.load_settings()
    name, _, _ = core.resolve_corpus(corpus)
    report = core.index_report(corpus=name)

    if not report["db_file"]:
        return _status_html("warn", "尚未建立索引", "上传文档后点击下方按钮建库"), _docs_html(name)
    n = report["active_chunks"] or 0
    if n == 0:
        return (_status_html("warn", f"{name} 无索引", "文档已就位，等待构建索引"),
                _docs_html(name))

    meta = (f"{n} 个 chunk · {core.describe_chunking(cfg)} · 精排 {core.rerank_label(cfg['rerank'])}"
            f" · 知识库 {name}")
    others = {k: v for k, v in report["by_corpus"].items() if k != name}
    if others:
        meta += "　其他：" + "、".join(f"{k} {v}" for k, v in others.items())
    if report["orphan_dirs"]:
        meta += f" · 待清理目录 {len(report['orphan_dirs'])} 个"
    return _status_html("ok", "索引已就绪", meta), _docs_html(name)


def _cfg_from_ui(chunk_size, chunk_overlap, adaptive, rerank, *, top_k=None,
                 candidate_k=None, rrf_k=None, bm25_weight=None,
                 vector_weight=None, rerank_model=None):
    """由界面控件值组装配置，**不读磁盘**。

    以前这里调 core.load_settings()，等于让「磁盘上的旧配置」和「界面上的新值」
    互相覆盖（改完下拉框又被读回来的旧值顶掉）。现在所有参数显式传入，
    没在界面上暴露的检索参数由调用方用 SEARCH_PARAMS 快照传进来。
    """
    cfg = dict(core.DEFAULTS)
    cfg.update({"chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap),
                "adaptive": bool(adaptive), "rerank": rerank or "off"})
    for key, val in (("top_k", top_k), ("candidate_k", candidate_k), ("rrf_k", rrf_k),
                     ("bm25_weight", bm25_weight), ("vector_weight", vector_weight),
                     ("rerank_model", rerank_model)):
        if val is not None:
            cfg[key] = val
    return core.normalize(cfg)


def estimate_chunks(chunk_size, chunk_overlap, adaptive, corpus=None):
    """实时预估分块结果（不调用 embedding，纯文本切分，毫秒级）"""
    try:
        cfg = _cfg_from_ui(chunk_size, chunk_overlap, adaptive, RERANK, **SEARCH_PARAMS)
        info = core.estimate_chunks(cfg, corpus=corpus)
    except Exception as e:
        return f'<div class="est est-warn">读取文档失败：{html.escape(str(e))}</div>'
    if not info["docs"]:
        return '<div class="est est-warn">这个知识库里还没有文档</div>'

    if adaptive:
        tags = "".join(
            f'<span class="est-tag">{html.escape(name)} · {cs}/{ov} → {n} 块</span>'
            for name, cs, ov, n in info["per_doc"])
        return (f'<div class="est">自适应：共 <b>{info["total"]}</b> 个 chunk'
                f'<div class="est-tags">{tags}</div></div>')

    return (f'<div class="est">预估 <b>{info["total"]}</b> 个 chunk'
            f'（{info["docs"]} 篇文档 · 平均每块 {info["avg"]:.0f} 字）</div>')


def rebuild_index(chunk_size, chunk_overlap, adaptive, rerank):
    """清空旧集合 → 清理孤儿目录 → 按当前参数重建全部知识库 → 预热检索器"""
    cfg = _cfg_from_ui(chunk_size, chunk_overlap, adaptive, rerank, **SEARCH_PARAMS)
    _drop_cache()

    try:
        stats = core.rebuild_all(cfg)
    except Exception as e:
        return (gr.update(value=f"⚠️ 重建失败：{e}"),
                *_status_html_wrapper("error", "建库失败", str(e)))

    if stats["chunks"] == 0:
        return (gr.update(value="⚠️ docs/ 下没有文档，请先上传"),
                *_status_html_wrapper("warn", "索引为空", "docs/ 里还没有文档"))

    # 只预热当前知识库，其余在切换时懒加载
    try:
        init_retriever(ACTIVE_CORPUS, cfg)
    except Exception as e:
        print(f"[app] 预热检索器失败：{e}")

    status, docs_view = get_status(ACTIVE_CORPUS)
    detail = "；".join(f"{name} {info['chunks']} 块"
                       for name, info in stats["corpora"].items())
    msg = (f"✅ 重建完成：{stats['docs']} 篇文档 → {stats['chunks']} 个 chunk"
           f"（{core.describe_chunking(cfg)}）　{detail}")
    if stats["purged"]:
        msg += f"　清理 {len(stats['purged'])} 个无用索引目录"
    if stats.get("purge_failed"):
        msg += f"　{len(stats['purge_failed'])} 个目录被占用，下次启动会自动再清"
    if stats.get("dropped"):
        msg += f"　删除过期集合 {', '.join(stats['dropped'])}"
    return gr.update(value=msg), status, docs_view


def clear_index(confirm, corpus=None):
    """只清空索引（collection + 残留目录），**不重建**。

    和「重建索引」拆成两个键：想先调切片参数再建库时，先清后建，
    避免旧索引口径（旧 chunk_size）和新参数混在一起。
    清理不可逆，所以要求先勾选确认框。
    """
    if not confirm:
        status, docs_view = get_status(corpus)
        return (gr.update(value="⚠️ 请先勾选下方「我确认清空全部索引」，再点清理"),
                status, docs_view)

    # 先丢掉手里的 Chroma 句柄，否则 Windows 会锁住目录删不掉
    _drop_cache()
    try:
        res = core.clear_all_index()
    except Exception as e:
        return (gr.update(value=f"⚠️ 清理失败：{e}"),
                _status_html("error", "清理失败", str(e)), _docs_html(corpus))

    status, docs_view = get_status(corpus)
    if not res["dropped"] and not res["purged"]:
        return gr.update(value="没有需要清理的索引（当前为空）"), status, docs_view

    msg = f"🧹 已清空 {len(res['dropped'])} 个集合"
    if res["purged"]:
        msg += f"，删除 {len(res['purged'])} 个残留目录"
    if res["purge_failed"]:
        msg += f"；{len(res['purge_failed'])} 个目录被占用，重启程序后会自动清理"
    if res["failed"]:
        msg += f"；{len(res['failed'])} 个删除失败"
    msg += "。切片参数已保留，随时可以点「重建索引」。"
    return gr.update(value=msg), status, docs_view


def apply_rerank(rerank, corpus=None):
    """切换精排方式：只换检索器里的精排器，不重新 embedding。

    依赖缺失（未装 sentence-transformers）时回退到 off，并且必须把
    「配置 / 检索器 / 下拉框」三处一起回填 —— 之前只返回了 gr.update，
    检索器内部还是旧精排器，界面显示与实际行为不一致。
    切片参数取自磁盘上「上次建库时」的配置，保证与索引口径一致（这里刻意不读界面值）。

    替换是**原子**的：先把新精排器全部造好，再统一挂上去。
    边造边换的话，中途抛错会留下「一部分检索器已换、一部分还是旧的」，
    而配置已经写盘，界面和实际行为再也对不上。
    """
    cfg = core.load_settings()
    fallback = None

    if rerank == "cross" and importlib.util.find_spec("sentence_transformers") is None:
        rerank = "off"
        fallback = "未安装 sentence-transformers，已回退为「关闭」"

    live = {name: r for name, r in _RETRIEVERS.items() if r is not None}

    # 1) 先造好（不改动任何现有对象）
    try:
        built = {name: (core.Reranker(rerank, cfg["rerank_model"])
                        if rerank != "off" else None)
                 for name in live}
    except Exception as e:
        status, _ = get_status(corpus)
        return (_status_html("error", "精排切换失败", f"{e}（仍保持 {cfg['rerank']}）"),
                gr.update(value=cfg["rerank"]))

    # 2) 再统一替换
    before = {name: r.reranker for name, r in live.items()}
    for name, r in live.items():
        r.reranker = built[name]

    # 3) 配置最后写；写失败就把检索器恢复原状
    try:
        cfg["rerank"] = rerank
        core.save_settings(cfg)
    except Exception as e:
        for name, r in live.items():
            r.reranker = before[name]
        return (_status_html("error", "精排切换失败", f"配置保存失败：{e}"),
                gr.update(value=core.load_settings()["rerank"]))
    for r in live.values():
        r.cfg["rerank"] = rerank

    status, _ = get_status(corpus)
    if fallback:
        status = _status_html("warn", "精排不可用", fallback)
    return status, gr.update(value=rerank)


def switch_corpus(corpus):
    """切换知识库：懒加载对应检索器，刷新状态卡"""
    try:
        get_retriever(corpus)
    except Exception as e:
        return (_status_html("error", "载入失败", f"{corpus}：{e}"),
                EMPTY_SOURCES, _docs_html(corpus))
    status, docs_view = get_status(corpus)
    return status, EMPTY_SOURCES, docs_view


def _status_html_wrapper(kind, title, meta):
    return _status_html(kind, title, meta), _docs_html()


def upload_files(files, corpus=None):
    """把上传的文件复制到当前知识库目录（docs/<corpus>/）"""
    if not files:
        return gr.update(value="没有选择文件"), *_status_html_wrapper("warn", "未上传", "请先选择文件")
    _, target_dir, _ = core.resolve_corpus(corpus)
    os.makedirs(target_dir, exist_ok=True)
    saved = []
    for f in files:
        src = f if isinstance(f, str) else getattr(f, "name", str(f))
        target = os.path.join(target_dir, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(target):
            shutil.copy(src, target)
        saved.append(os.path.basename(src))
    status, docs_view = get_status(corpus)
    where = os.path.basename(target_dir) or "docs"
    return (gr.update(value=f"✅ 已上传 {len(saved)} 个文件到 {where}/："
                            f"{', '.join(saved)}　→ 记得点击「重建索引」"),
            status, docs_view)


# ------------------------- 交互逻辑 -------------------------
def _render_sources(rows, elapsed=None):
    """rows: [(doc, meta)]，meta 里带 BM25 / 向量 / 精排 的名次"""
    if not rows:
        return EMPTY_SOURCES
    cards = []
    for i, (d, meta) in enumerate(rows, 1):
        src = os.path.basename(str(d.metadata.get("source", "未知来源")))
        text = html.escape(d.page_content.strip()).replace("\n", "<br>")
        if len(text) > 320:
            text = text[:320] + "…"
        tags = []
        if meta.get("bm25_rank"):
            tags.append(f"BM25 #{meta['bm25_rank']}")
        if meta.get("vector_rank"):
            tags.append(f"向量 #{meta['vector_rank']}")
        if meta.get("rerank_rank"):
            tags.append(f"精排 #{meta['rerank_rank']}")
        tags_html = "".join(f'<span class="src-tag">{t}</span>' for t in tags)
        cards.append(
            f'<div class="src-item">'
            f'<div class="src-head"><span class="src-rank">{i}</span>'
            f'<span class="src-name">{html.escape(src)}</span>{tags_html}</div>'
            f'<div class="src-body">{text}</div></div>'
        )
    head = f'<div class="src-meta">命中 {len(rows)} 个片段'
    if elapsed is not None:
        head += f' · 检索 {elapsed:.2f}s'
    head += "</div>"
    return head + "".join(cards)


def respond(message, history, corpus=None):
    """流式问答（复用缓存检索器，不再每次新建连接）
    关键：先 yield 一次把用户提问和"检索中"占位立即推到界面，
    否则要等检索 + 首 token（约 10s）才刷新，看起来像按钮没反应。
    """
    message = (message or "").strip()
    if not message:
        return

    history = [dict(m) for m in (history or [])]
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "🔍 正在检索知识库…"})
    yield history, "", EMPTY_SOURCES  # ← 立即回显，不等检索

    retriever = get_retriever(corpus)
    if retriever is None:
        history[-1] = {"role": "assistant",
                       "content": "⚠️ 这个知识库还没建好索引。请上传文档后点击「重建索引」。"}
        yield history, "", EMPTY_SOURCES
        return

    t0 = time.time()
    try:
        rows = retriever.invoke_with_meta(message)
    except Exception as e:
        history[-1] = {"role": "assistant",
                       "content": f"⚠️ 检索失败：{e}\n\n请确认 Ollama 已启动（`ollama serve`）。"}
        yield history, "", EMPTY_SOURCES
        return

    docs = [d for d, _ in rows]
    context = core.format_docs(docs)
    messages = PROMPT.format_messages(context=context, question=message)
    sources_html = _render_sources(rows, time.time() - t0)

    history[-1] = {"role": "assistant", "content": "✍️ 正在生成…"}
    yield history, "", sources_html  # ← 检索完成，先把来源推上去

    partial = ""
    history[-1] = {"role": "assistant", "content": "▌"}
    try:
        for chunk in llm.stream(messages):
            partial += chunk.content or ""
            history[-1] = {"role": "assistant", "content": partial + "▌"}
            yield history, "", sources_html
    except Exception as e:
        history[-1] = {"role": "assistant",
                       "content": f"⚠️ 生成失败：{e}\n\n请确认 Ollama 已启动（`ollama serve`）。"}
        yield history, "", sources_html
        return

    history[-1] = {"role": "assistant",
                   "content": partial.strip() or "（模型没有返回内容，请重试）"}
    yield history, "", sources_html


def clear_chat():
    return WELCOME, "", EMPTY_SOURCES


# ------------------------- 样式 -------------------------
CSS = """
:root {
  --ink: #1b2130;
  --muted: #6b7280;
  --line: #e8eaf3;
  --brand: #5b5bd6;
  --page: #f4f5fa;
}
html, body { background: var(--page) !important; }

.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
  padding: 30px 30px 18px !important;
  background:
    radial-gradient(900px 420px at 6% -10%, #e9ecff 0%, rgba(233,236,255,0) 62%),
    radial-gradient(820px 400px at 98% -6%, #ffeaf3 0%, rgba(255,234,243,0) 58%),
    var(--page) !important;
  font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important;
  color: var(--ink) !important;
}
footer { display: none !important; }

/* ============================================================
   立方体舞台：顶面（透视梯形）+ 正面（矩形主体）+ 底托
   ============================================================ */
.cube-stage {
  position: relative !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* ---------- 顶面 ---------- */
.face-top {
  position: relative;
  z-index: 3;
  padding: 0 !important;
  gap: 0 !important;
  background: transparent !important;
  border: none !important;
  filter: drop-shadow(0 16px 24px rgba(91, 91, 214, .26));
}
.face-top > *, .face-top > * > * {
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  border: none !important;
}
.hero {
  position: relative;
  overflow: hidden;
  width: 100% !important;
  box-sizing: border-box !important;
  margin: 0 !important;
  /* 上窄下宽 = 从斜上方看立方体的顶面 */
  clip-path: polygon(1.6% 0%, 98.4% 0%, 100% 100%, 0% 100%);
  border-radius: 0;
  padding: 22px 36px 30px;
  color: #fff;
  background:
    linear-gradient(180deg, rgba(255,255,255,.16) 0%, rgba(255,255,255,0) 46%),
    linear-gradient(112deg, #5b5bd6 0%, #7c5cf0 46%, #c2559e 100%);
}
.hero::after {
  content: "";
  position: absolute; right: -70px; top: -90px;
  width: 300px; height: 300px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,.30) 0%, rgba(255,255,255,0) 68%);
}
/* 顶面与正面的折角：底部压一层暗色，读作立方体的转折 */
.hero::before {
  content: "";
  position: absolute; left: 0; right: 0; bottom: 0;
  height: 26px;
  pointer-events: none;
  background: linear-gradient(180deg, rgba(44,28,96,0) 0%, rgba(44,28,96,.30) 100%);
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11.5px; letter-spacing: .08em; text-transform: uppercase;
  padding: 4px 11px; border-radius: 999px;
  background: rgba(255,255,255,.18);
  border: 1px solid rgba(255,255,255,.30);
  backdrop-filter: blur(6px);
}
.hero h1 {
  margin: 11px 0 6px; font-size: 26px; font-weight: 700;
  letter-spacing: -.02em; line-height: 1.2;
}
.hero p { margin: 0; font-size: 13.5px; opacity: .92; }
.hero-chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 13px; }
.hero-chips span {
  font-size: 12px; padding: 4px 10px; border-radius: 999px;
  background: rgba(255,255,255,.16);
  border: 1px solid rgba(255,255,255,.26);
}

/* ---------- 正面 ---------- */
.face-front {
  position: relative;
  z-index: 2;
  display: flex !important;
  flex-direction: column !important;
  gap: 0 !important;
  margin: -14px 0 0 !important;
  padding: 26px 16px 16px !important;
  background: #fff !important;
  border: 1px solid var(--line) !important;
  border-radius: 20px !important;
  box-shadow:
    inset 10px 0 16px -14px rgba(96, 102, 140, .55),
    inset -10px 0 16px -14px rgba(96, 102, 140, .55),
    inset 0 1px 0 rgba(255, 255, 255, .9),
    0 34px 60px -36px rgba(23, 28, 45, .42) !important;
}
/* 立方体左右侧面（厚度） */
.face-front::before,
.face-front::after {
  content: "";
  position: absolute;
  top: 0; bottom: 0;
  width: 11px;
  pointer-events: none;
}
.face-front::before {
  left: 0;
  background: linear-gradient(90deg, #e3e5f3 0%, rgba(227,229,243,0) 100%);
}
.face-front::after {
  right: 0;
  background: linear-gradient(270deg, #e3e5f3 0%, rgba(227,229,243,0) 100%);
}

/* ---------- 底托（底边收窄 = 透视） ---------- */
.floor-slot {
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  border: none !important;
}
.cube-floor {
  height: 20px;
  margin: -7px 14px 0;
  clip-path: polygon(0% 0%, 100% 0%, 97.3% 100%, 2.7% 100%);
  background: linear-gradient(180deg, #d7daee 0%, rgba(215,218,238,.5) 52%, rgba(215,218,238,0) 100%);
  filter: blur(.4px);
}

/* ---------- 正面内的两栏 ---------- */
.front-split {
  display: flex !important;
  flex-wrap: nowrap !important;
  align-items: stretch !important;
  gap: 16px !important;
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  border: none !important;
}
.panel-left {
  padding: 0 4px 0 2px !important;
  gap: 10px !important;
  background: transparent !important;
  border: none !important;
  min-width: 0 !important;
}
.panel-right {
  padding: 14px 14px 12px !important;
  gap: 6px !important;
  background: #fafbff !important;
  border: 1px solid #eff1fa !important;
  border-radius: 14px !important;
  min-width: 0 !important;
}
.card-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 2px; }
.card-title { font-size: 15px; font-weight: 650; color: var(--ink); }
.card-sub { font-size: 12.5px; color: var(--muted); }

/* ---------- 聊天区 ---------- */
#chatbot {
  border: none !important;
  background: transparent !important;
  padding: 0 !important;
}
#chatbot .message-row .message,
#chatbot .message { box-shadow: none !important; }
#chatbot .message-row .bubble,
#chatbot .bubble {
  border-radius: 15px !important;
  box-shadow: none !important;
  font-size: 14.5px !important;
  line-height: 1.68 !important;
  padding: 11px 15px !important;
}
#chatbot .message-row.user-row .bubble,
#chatbot .message.user .bubble {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
  color: #fff !important;
  border: none !important;
}
#chatbot .message-row.bot-row .bubble,
#chatbot .message.bot .bubble {
  background: #f4f5fb !important;
  color: #1f2430 !important;
  border: 1px solid #eceef7 !important;
}

/* 隐藏消息右上角的悬浮操作图标，保持画面干净 */
#chatbot .message-buttons,
#chatbot .icon-button-wrapper,
#chatbot .message-row .icon-buttons,
#chatbot button.icon-button {
  display: none !important;
}

/* 输入区 */
#msg-box {
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  background: #fff !important;
  box-shadow: none !important;
  padding: 2px 4px !important;
}
#msg-box textarea {
  border: none !important;
  box-shadow: none !important;
  font-size: 14.5px !important;
  padding: 12px 8px !important;
  background: transparent !important;
}
#msg-box textarea:focus { outline: none !important; }
#send-btn {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  color: #fff !important; border: none !important;
  border-radius: 13px !important; font-weight: 600 !important;
  min-width: 84px !important; box-shadow: 0 8px 18px -8px rgba(99,102,241,.75) !important;
}
#send-btn:hover { filter: brightness(1.05); }
#clear-btn {
  background: #fff !important; color: var(--muted) !important;
  border: 1px solid var(--line) !important;
  border-radius: 13px !important; font-weight: 600 !important;
  min-width: 68px !important;
}
#clear-btn:hover { color: #b4443f !important; border-color: #f3cfcd !important; }

/* 示例问题 */
.examples-wrap { margin-top: 4px; }
.examples-wrap td, .examples-wrap .gallery-item { border-radius: 12px !important; }

/* ---------- 参考片段 ---------- */
#sources-acc {
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  background: #fcfcff !important;
  margin-top: 6px !important;
}
#sources-acc summary { font-size: 13.5px !important; font-weight: 600 !important; color: #4b5563 !important; }
.src-meta { font-size: 12.5px; color: var(--muted); margin: 2px 0 10px; }
.src-empty {
  display: flex; align-items: center; gap: 10px;
  font-size: 13px; color: #8b90a3; padding: 14px 6px;
}
.src-empty-icon { font-size: 16px; }
.src-item {
  border: 1px solid #edeff7; border-radius: 12px;
  padding: 10px 12px; margin-bottom: 8px; background: #fff;
}
.src-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.src-rank {
  width: 20px; height: 20px; border-radius: 7px; flex: none;
  background: #eef0ff; color: #5b5bd6;
  font-size: 11.5px; font-weight: 700;
  display: inline-flex; align-items: center; justify-content: center;
}
.src-name { font-size: 12.5px; font-weight: 600; color: #4b5563; }
.src-tag {
  font-size: 10.5px; color: #7c83a8; background: #f2f3fb;
  border: 1px solid #e7e9f6; border-radius: 5px; padding: 1px 5px;
  margin-left: 4px; white-space: nowrap;
}
.src-body { font-size: 12.5px; line-height: 1.72; color: #5b6273; }

/* ---------- 状态 ---------- */
.status {
  display: flex; align-items: center; justify-content: space-between;
  border-radius: 14px; padding: 12px 14px; border: 1px solid transparent;
}
.status-ok { background: #f0fbf5; border-color: #d6f2e3; }
.status-warn { background: #fffaf0; border-color: #fbe8c8; }
.status-error { background: #fef3f2; border-color: #fad9d6; }
.status-left { display: flex; align-items: center; gap: 11px; }
.status-dot {
  width: 26px; height: 26px; border-radius: 50%; flex: none;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700;
}
.status-ok .status-dot { background: #d8f5e5; color: #12855a; }
.status-warn .status-dot { background: #fdedd2; color: #a86b12; }
.status-error .status-dot { background: #fbdedb; color: #b4443f; }
.status-title { font-size: 13.5px; font-weight: 650; color: #24303f; }
.status-meta { font-size: 12px; color: #7b8494; margin-top: 1px; }

/* ---------- 文档清单 ---------- */
.doc-list {
  display: flex; flex-direction: column; gap: 6px;
  max-height: 118px; overflow-y: auto; padding-right: 2px;
}
.doc-list::-webkit-scrollbar { width: 6px; }
.doc-list::-webkit-scrollbar-thumb { background: #dcdfee; border-radius: 3px; }
.doc-group {
  font-size: 11px; font-weight: 600; letter-spacing: .5px;
  color: #8b90a6; text-transform: uppercase;
  margin: 8px 0 4px;
}
.doc-group:first-child { margin-top: 0; }
.doc-item {
  display: flex; align-items: center; gap: 9px;
  font-size: 12.8px; color: #414b5a;
  background: #fbfbfe; border: 1px solid #eef0f7;
  border-radius: 10px; padding: 6px 9px;
}
.doc-icon { font-size: 13px; }
.doc-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-size { color: #9aa1b1; font-size: 11.5px; flex: none; }
.doc-empty { font-size: 12.5px; color: #9aa1b1; padding: 10px 2px; }

/* ---------- 正面底部：功能区（平铺三栏） ---------- */
.face-base {
  display: flex !important;
  flex-wrap: wrap !important;
  align-items: stretch !important;
  gap: 12px !important;
  margin: 16px -16px 0 !important;
  padding: 14px 16px 0 !important;
  border-top: 1px dashed #e3e6f5 !important;
  background: linear-gradient(180deg, rgba(250,251,255,0) 0%, rgba(242,243,252,.92) 100%) !important;
  border-radius: 0 0 18px 18px !important;
}
.base-cell {
  flex: 1 1 250px !important;
  height: 100% !important;
  box-sizing: border-box !important;
  padding: 11px 13px !important;
  gap: 5px !important;
  background: #fff !important;
  border: 1px solid #eaecf8 !important;
  border-radius: 13px !important;
  min-width: 0 !important;
  align-self: stretch !important;
  box-shadow: 0 6px 14px -12px rgba(30, 36, 60, .5) !important;
}
.base-cell > * { margin: 0 !important; }
.cell-note {
  font-size: 11.2px; color: #a0a6b6; line-height: 1.55; margin-top: 1px;
}
.cell-head {
  display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
  margin-bottom: 2px;
}
.cell-title { font-size: 13px; font-weight: 650; color: var(--ink); }
.cell-sub { font-size: 11.3px; color: #9aa1b1; white-space: nowrap; }

/* ---------- 上传区 ---------- */
#upload-zone {
  border: 1.5px dashed #cfd4ee !important;
  border-radius: 12px !important;
  background: #fbfbff !important;
  min-height: 88px !important;
}
#upload-zone:hover { border-color: #a5a8f0 !important; background: #f7f7ff !important; }
#upload-btn {
  background: #fff !important; color: #4b4bd0 !important;
  border: 1px solid #d9dcf5 !important;
  border-radius: 11px !important; font-weight: 620 !important;
  min-height: 38px !important; font-size: 13px !important;
}
#upload-btn:hover { background: #f5f5ff !important; }
#rebuild-btn {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  color: #fff !important; border: none !important;
  border-radius: 11px !important; font-weight: 650 !important;
  min-height: 38px !important; font-size: 13px !important;
  box-shadow: 0 10px 20px -12px rgba(99,102,241,.9) !important;
}
#rebuild-btn:hover { filter: brightness(1.05); }
#purge-btn {
  background: #fff !important; color: #b4443f !important;
  border: 1px solid #f3cfcd !important;
  border-radius: 11px !important; font-weight: 620 !important;
  min-height: 38px !important; font-size: 13px !important;
}
#purge-btn:hover { background: #fef3f2 !important; border-color: #eab8b4 !important; }

/* 侧栏小节标题 */
.side-title {
  font-size: 11.5px; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; color: #9aa1b1; margin: 4px 0 6px;
}
.result-md { font-size: 12px !important; color: #5b6273 !important; line-height: 1.6 !important; }
.result-md p { margin: 3px 0 !important; }

/* ---------- 切片参数 ---------- */
.est {
  font-size: 12.2px; color: #5b6273; background: #f7f8ff;
  border: 1px dashed #e2e5f5; border-radius: 10px;
  padding: 8px 10px; margin-top: 2px;
}
.est b { color: #4b4bd0; }
.est-warn { color: #a86b12 !important; background: #fffaf0 !important; border-color: #fbe8c8 !important; }
.est-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
.est-tag {
  font-size: 11px; background: #fff; border: 1px solid #e6e8f5;
  border-radius: 6px; padding: 2px 6px; color: #6b7280;
}
#cs-slider label, #ov-slider label { font-size: 12.5px !important; font-weight: 600 !important; }
/* 自定义滑块标题行（避开 Gradio 自带 label 的渲染问题） */
.ctl-row {
  display: flex; align-items: baseline; justify-content: space-between;
  margin: 10px 0 2px; font-size: 12.5px; font-weight: 650; color: #374151;
}
.ctl-row code {
  font-size: 11.3px; font-weight: 500; color: #9aa1b1;
  background: transparent; padding: 0;
}
.ctl-hint { font-size: 11.3px; font-weight: 500; color: #9aa1b1; }
#cs-slider, #ov-slider, #adap-cb,
#cs-slider *, #ov-slider *, #adap-cb * {
  -webkit-user-select: none !important;
  user-select: none !important;
}
#cs-slider ::selection, #ov-slider ::selection, #adap-cb ::selection {
  background: transparent !important;
}
#cs-slider label, #ov-slider label, #adap-cb label {
  background: transparent !important;
  color: #374151 !important;
}
#cs-slider .info, #ov-slider .info, #adap-cb .info { font-size: 11.5px !important; }
/* Gradio 的 checkbox 文字默认用主色，统一成深灰 */
#adap-cb .wrap, #adap-cb span, #adap-cb label { color: #374151 !important; }
#rerank-dd {
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  background: #fff !important;
  min-height: 36px !important;
  font-size: 12.5px !important;
}
#rerank-dd input { font-size: 12.5px !important; color: #374151 !important; }

/* 危险操作确认勾选 */
#confirm-cb { margin-top: 2px !important; }
#confirm-cb label, #confirm-cb span { font-size: 11.8px !important; color: #8a6b6b !important; }
#confirm-cb input { accent-color: #d9534f !important; }

/* 底部 */
.foot {
  text-align: center; font-size: 12px; color: #a3a9b8;
  padding: 8px 0 4px;
}

/* ============================================================
   响应式：窄屏收起透视，优先保证可读与可点
   ============================================================ */
@media (max-width: 1080px) {
  .gradio-container { padding: 22px 16px 14px !important; }
  .hero {
    clip-path: none;
    border-radius: 18px 18px 0 0;
    padding: 20px 22px 24px;
    margin-bottom: -1px !important;
  }
  .face-top { filter: none; }
  .face-front {
    margin-top: 0 !important;
    border-radius: 0 0 18px 18px !important;
    padding: 18px 12px 12px !important;
  }
  .face-front::before,
  .face-front::after,
  .cube-floor { display: none !important; }
  .front-split { flex-wrap: wrap !important; }
  .front-split > * { width: 100% !important; flex: 1 1 100% !important; }
  .face-base { margin: 14px 0 0 !important; padding: 14px 0 0 !important; }
  .face-base > * { flex: 1 1 calc(50% - 6px) !important; }
  .face-base > *:last-child { flex: 1 1 100% !important; }
}
@media (max-width: 720px) {
  .gradio-container { padding: 14px 10px 10px !important; }
  .hero { padding: 16px 16px 20px; }
  .hero h1 { font-size: 21px; }
  .hero-chips { display: none; }
  .face-front { padding: 14px 8px 10px !important; border-radius: 0 0 14px 14px !important; }
  .base-cell { flex: 1 1 100% !important; }
  .face-base { border-radius: 0 !important; }
}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    radius_size="lg",
    font=["-apple-system", "Segoe UI", "PingFang SC", "Microsoft YaHei", "sans-serif"],
)

HERO = """
<div class="hero">
  <div class="hero-badge">RAG · 检索增强生成</div>
  <h1>📚 RAG 智能助手</h1>
  <p>基于本地文档的问答系统 —— 答案只来自你的知识库，不知道就明确说不知道。</p>
  <div class="hero-chips">
    <span>LangChain</span><span>Chroma</span><span>Ollama · qwen2.5</span>
    <span>bge-m3</span><span>BM25 + 向量 RRF 融合</span>
  </div>
</div>
"""

EXAMPLES = [
    "学生违纪处分分为哪几个等级？",
    "考试找他人替考会受到什么处分？",
    "宿舍管理有哪两条红线？",
    "学生申诉由哪个机构受理？",
]

# —— 启动预热：先清孤儿目录（此时没有打开的 Chroma 连接，最容易删干净）→ 初始化默认知识库 ——
if os.path.exists(DB_DIR):
    try:
        _purged = core.purge_orphan_segments()
        if _purged["failed"]:
            print(f"启动清理：{len(_purged['failed'])} 个索引目录被占用，跳过")
    except Exception as _e:
        print(f"启动清理孤儿目录失败：{_e}")
    try:
        init_retriever(ACTIVE_CORPUS)
    except Exception as _e:
        print(f"启动时索引初始化失败：{_e}")
try:
    INIT_STATUS, INIT_DOCS = get_status(ACTIVE_CORPUS)
except Exception as _e:
    INIT_STATUS, INIT_DOCS = _status_html("error", "状态读取失败", str(_e)), _docs_html()

INIT_ESTIMATE = estimate_chunks(CHUNK_SIZE, CHUNK_OVERLAP, ADAPTIVE, ACTIVE_CORPUS)


def _cell_head(icon, title, sub):
    return (f'<div class="cell-head"><span class="cell-title">{icon} {title}</span>'
            f'<span class="cell-sub">{sub}</span></div>')


# —— Gradio 界面布局：顶面（Hero）+ 正面（对话/知识库）+ 正面下方平铺功能区 ——
with gr.Blocks(title="RAG 智能助手", fill_width=True) as demo:
    with gr.Column(elem_classes="cube-stage"):
        # ================= 立方体顶面 =================
        with gr.Column(elem_classes="face-top"):
            gr.HTML(HERO)

        # ================= 立方体正面 =================
        with gr.Column(elem_classes="face-front"):
            with gr.Row(elem_classes="front-split", equal_height=False):
                # ---------- 左：对话 ----------
                with gr.Column(scale=7, elem_classes="panel-left"):
                    gr.HTML("""
                    <div class="card-head">
                      <span class="card-title">💬 对话</span>
                      <span class="card-sub">仅依据知识库作答 · 支持流式输出</span>
                    </div>
                    """)

                    chatbot = gr.Chatbot(
                        value=WELCOME,
                        elem_id="chatbot",
                        show_label=False,
                        height=372,
                        buttons=[],
                    )

                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="输入你的问题，回车发送…",
                            show_label=False,
                            container=False,
                            submit_btn=False,
                            autofocus=True,
                            scale=9,
                            elem_id="msg-box",
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1, elem_id="send-btn")
                        clear_btn = gr.Button("清空", variant="secondary", scale=1, elem_id="clear-btn")

                    with gr.Accordion("🔍 参考片段（答案来源）", open=False, elem_id="sources-acc"):
                        sources = gr.HTML(EMPTY_SOURCES)

                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=msg_box,
                        label="试试这样问",
                        examples_per_page=4,
                    )

                # ---------- 右：知识库 ----------
                with gr.Column(scale=4, elem_classes="panel-right"):
                    gr.HTML('<div class="ctl-row" style="margin-top:0">'
                            '<span>当前知识库</span>'
                            '<span class="ctl-hint">docs/ 子目录各成一体</span></div>')
                    corpus_dd = gr.Dropdown(
                        choices=CORPORA, value=ACTIVE_CORPUS,
                        show_label=False, container=False, elem_id="corpus-dd",
                    )

                    gr.HTML('<div class="side-title">知识库状态</div>')
                    status = gr.HTML(INIT_STATUS)

                    gr.HTML('<div class="side-title">文档列表</div>')
                    docs_view = gr.HTML(INIT_DOCS)

                    gr.HTML("""
                    <div class="side-title">工作原理</div>
                    <div style="font-size:12.2px;line-height:1.85;color:#6b7280">
                      ① 递归分块（按下方切片参数）<br>
                      ② bge-m3 向量化入 Chroma<br>
                      ③ BM25 + 向量双路召回<br>
                      ④ RRF 融合 → 可选精排<br>
                      ⑤ qwen2.5:7b 受约束生成
                    </div>
                    """)

            # ---------- 正面下方：平铺功能区 ----------
            with gr.Row(elem_classes="face-base", equal_height=True):
                # ① 上传
                with gr.Column(scale=10, elem_classes="base-cell"):
                    gr.HTML(_cell_head("📤", "上传文档", "存入当前知识库"))
                    file_input = gr.File(
                        label="上传文档",
                        show_label=False,
                        file_types=[".pdf", ".md", ".txt"],
                        file_count="multiple",
                        elem_id="upload-zone",
                    )
                    upload_btn = gr.Button("上传到当前知识库", variant="secondary",
                                           elem_id="upload-btn")
                    gr.HTML('<div class="cell-note">支持 PDF / Markdown / TXT，可多个一起选</div>')
                    upload_result = gr.Markdown(elem_classes="result-md")

                # ② 切片参数
                with gr.Column(scale=12, elem_classes="base-cell"):
                    gr.HTML(_cell_head("⚙️", "切片参数", "改动后需重建"))
                    gr.HTML('<div class="ctl-row"><span>分块大小 <code>chunk_size</code></span>'
                            '<span class="ctl-hint">条文类建议 800+</span></div>')
                    cs_slider = gr.Slider(
                        minimum=300, maximum=1500, step=50, value=CHUNK_SIZE,
                        show_label=False, container=False,
                        interactive=not ADAPTIVE, elem_id="cs-slider",
                    )
                    gr.HTML('<div class="ctl-row"><span>重叠长度 <code>overlap</code></span>'
                            '<span class="ctl-hint">建议 10%~20%</span></div>')
                    ov_slider = gr.Slider(
                        minimum=0, maximum=300, step=10, value=CHUNK_OVERLAP,
                        show_label=False, container=False,
                        interactive=not ADAPTIVE, elem_id="ov-slider",
                    )
                    adaptive_cb = gr.Checkbox(
                        value=ADAPTIVE,
                        label="自适应分块（按文档长度自动定参）",
                        elem_id="adap-cb",
                    )
                    estimate = gr.HTML(INIT_ESTIMATE)

                # ③ 索引与检索
                with gr.Column(scale=11, elem_classes="base-cell"):
                    gr.HTML(_cell_head("🗂", "索引与检索",
                                       f"{len(CORPORA)} 个知识库"))
                    gr.HTML('<div class="ctl-row"><span>精排 reranker</span>'
                            '<span class="ctl-hint">改动立即生效</span></div>')
                    rerank_dd = gr.Dropdown(
                        choices=RERANK_CHOICES, value=RERANK,
                        show_label=False, container=False, elem_id="rerank-dd",
                    )
                    with gr.Row():
                        purge_btn = gr.Button("🧹 清理旧索引", variant="secondary",
                                              elem_id="purge-btn")
                        rebuild_btn = gr.Button("🔄 重建索引", variant="primary",
                                                elem_id="rebuild-btn")
                    gr.HTML('<div class="cell-note">清理只删索引、不动文档；'
                            '重建按当前切片参数全量重算</div>')
                    confirm_cb = gr.Checkbox(
                        value=False, label="我确认清空全部索引（不可恢复）",
                        elem_id="confirm-cb",
                    )
                    index_result = gr.Markdown(elem_classes="result-md")

        gr.HTML('<div class="cube-floor"></div>', elem_classes="floor-slot")

    gr.HTML('<div class="foot">本地运行 · 数据不出电脑 · LangChain + Chroma + Ollama</div>')

    # —— 事件绑定 ——
    submit_args = dict(fn=respond, inputs=[msg_box, chatbot, corpus_dd],
                       outputs=[chatbot, msg_box, sources], show_progress="hidden")
    msg_box.submit(**submit_args, api_name="ask")
    send_btn.click(**submit_args, api_name=False)

    clear_btn.click(fn=clear_chat, inputs=None,
                    outputs=[chatbot, msg_box, sources], api_name=False)

    upload_btn.click(fn=upload_files, inputs=[file_input, corpus_dd],
                     outputs=[upload_result, status, docs_view])

    # 两个独立动作：清空旧索引（不重建） / 按当前参数重建
    purge_btn.click(fn=clear_index, inputs=[confirm_cb, corpus_dd],
                    outputs=[index_result, status, docs_view]).then(
        fn=lambda: gr.update(value=False), inputs=None, outputs=confirm_cb,
        show_progress="hidden",
    )

    rebuild_btn.click(fn=rebuild_index, inputs=[cs_slider, ov_slider, adaptive_cb, rerank_dd],
                      outputs=[index_result, status, docs_view]).then(
        fn=estimate_chunks, inputs=[cs_slider, ov_slider, adaptive_cb, corpus_dd],
        outputs=estimate
    )

    # 精排切换 → 立刻换检索器（不必重建索引）
    rerank_dd.change(fn=apply_rerank, inputs=[rerank_dd, corpus_dd],
                     outputs=[status, rerank_dd], show_progress="hidden")

    # 切换知识库 → 懒加载检索器 + 刷新状态与文档清单
    corpus_dd.change(fn=switch_corpus, inputs=corpus_dd,
                     outputs=[status, sources, docs_view], show_progress="hidden")

    # 切片参数变化 → 实时预估 chunk 数
    for comp in (cs_slider, ov_slider, adaptive_cb):
        comp.change(fn=estimate_chunks, inputs=[cs_slider, ov_slider, adaptive_cb, corpus_dd],
                    outputs=estimate, show_progress="hidden")

    # 勾选自适应时，禁用两个手动滑块
    adaptive_cb.change(
        fn=lambda a: (gr.update(interactive=not a), gr.update(interactive=not a)),
        inputs=adaptive_cb, outputs=[cs_slider, ov_slider], show_progress="hidden",
    )


if __name__ == "__main__":
    print("启动中... 浏览器打开 http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860,
                inbrowser=False, quiet=True, css=CSS, theme=THEME,
                footer_links=[])
