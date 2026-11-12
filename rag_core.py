"""
rag_core.py —— 01 / 02 / 03 / 04 共用的检索内核（唯一事实来源）

为什么要抽出这个文件：
  之前 01、02、04 各自写了一份「加载 → 分块」逻辑，参数还会不一致
  （02_query.py 把 chunk_size 硬编码成 800/120，UI 改成 400/60 重建后，
   CLI 的 BM25 与向量库分块口径就错位了，03_eval 也跟着失真）。
  现在分块参数、混合检索、reranker、索引维护全部收敛到这里，四个入口共用。

包含：
  - 切片配置读写（rag_ui_config.json）
  - 多知识库：docs/ 下每个子目录算一个语料（school / company …），各建一个 collection，
    避免两个领域的文档混在一个索引里互相干扰；根目录散文件只在没有子目录时才回退成 default
  - 文档加载 / 固定分块 / 自适应分块，并给每个 chunk 写稳定 ID
  - BM25 + 向量 RRF 融合检索（按稳定 ID 去重，不再用前 100 字符当 key）
    RRF 的 k 可配置（rrf_k），小语料默认 20，避免排序过于平坦
  - 可选 reranker（off / embed / llm / cross）
  - 索引重建与孤儿目录清理（Chroma 删集合不删 HNSW 目录）
  - index_report 带缓存（以 sqlite 的 mtime+size 为版本号）
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys


def _force_utf8_stdio():
    """把 stdout/stderr 切到 UTF-8。

    Windows 控制台默认 cp936(GBK)，脚本里 print 一个 ✅ / ✗ 就
    `UnicodeEncodeError: 'gbk' codec can't encode character` 直接崩，
    而且崩的位置往往在流程末尾（打印汇总时），前面的工作全白做。
    reconfigure 只对 TextIOWrapper 有效（重定向到管道、IPython 的 stdout
    是别的类型），所以这里静默跳过，并留 errors="replace" 兜底。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdio()

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------- 路径与默认配置 -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_DIR = os.path.join(BASE_DIR, "vectorstore")
CONFIG_PATH = os.path.join(BASE_DIR, "rag_ui_config.json")

COLLECTION = "demo"          # 兼容旧代码；多知识库下实际 collection 名由 corpus 决定
DEFAULT_CORPUS = "default"   # docs/ 根目录下的散文件归属
EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:7b"

SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", " ", ""]
# 自适应分块：(文档长度上限, chunk_size, overlap)
ADAPTIVE_TIERS = [(2000, 400, 60), (6000, 700, 110), (10 ** 9, 1000, 160)]

DEFAULTS = {
    "chunk_size": 800,
    "chunk_overlap": 120,
    "adaptive": False,
    "top_k": 4,          # 最终喂给 LLM 的片段数
    "candidate_k": 12,   # 每路召回候选数（精排从这里面挑）
    "bm25_weight": 0.4,
    "vector_weight": 0.6,
    "rrf_k": 20,         # RRF 公式里的 k：越小越偏重头部排名；小语料用 20（原来硬编码 60 太扁平）
    "rerank": "off",     # off / embed / llm / cross
    "rerank_model": LLM_MODEL,
}

RERANK_BACKENDS = ("off", "embed", "llm", "cross")
RERANK_LABELS = {
    "off": "关闭",
    "embed": "向量重排",
    "llm": "LLM 精排",
    "cross": "bge-reranker",
}


def rerank_label(backend: str) -> str:
    return RERANK_LABELS.get(backend, backend)


# ------------------------- 配置 -------------------------
def load_settings(path: str = CONFIG_PATH) -> dict:
    """读取切片/检索配置，缺失字段用默认值补齐并做合法性约束"""
    cfg = dict(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for key in ("chunk_size", "chunk_overlap", "top_k", "candidate_k", "rrf_k"):
            if key in raw:
                cfg[key] = int(raw[key])
        for key in ("bm25_weight", "vector_weight"):
            if key in raw:
                cfg[key] = float(raw[key])
        cfg["adaptive"] = bool(raw.get("adaptive", cfg["adaptive"]))
        if raw.get("rerank") in RERANK_BACKENDS:
            cfg["rerank"] = raw["rerank"]
        if raw.get("rerank_model"):
            cfg["rerank_model"] = str(raw["rerank_model"])
    except FileNotFoundError:
        pass
    except Exception as e:  # 配置损坏不该让程序起不来
        print(f"[rag_core] 读取配置失败，使用默认值：{e}")
    return normalize(cfg)


def normalize(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["chunk_size"] = max(100, int(cfg["chunk_size"]))
    cfg["chunk_overlap"] = max(0, min(int(cfg["chunk_overlap"]), cfg["chunk_size"] - 50))
    cfg["top_k"] = max(1, int(cfg["top_k"]))
    cfg["candidate_k"] = max(cfg["top_k"], int(cfg["candidate_k"]))
    cfg["rrf_k"] = max(1, int(cfg["rrf_k"]))
    return cfg


def save_settings(cfg: dict, path: str = CONFIG_PATH) -> dict:
    cfg = normalize(cfg)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[rag_core] 保存配置失败：{e}")
    return cfg


def describe_chunking(cfg: dict) -> str:
    if cfg.get("adaptive"):
        return "自适应分块"
    return f"分块 {cfg['chunk_size']}/{cfg['chunk_overlap']}"


# ------------------------- 多知识库（语料） -------------------------
DOC_EXTS = (".pdf", ".md", ".txt")


def _has_docs(path: str, direct_only: bool = False) -> bool:
    """目录里是否有可入库的文档（direct_only=True 时只看当前层）"""
    if not os.path.isdir(path):
        return False
    if direct_only:
        return any(f.lower().endswith(DOC_EXTS) and os.path.isfile(os.path.join(path, f))
                   for f in os.listdir(path))
    for root, _, files in os.walk(path):
        if any(f.lower().endswith(DOC_EXTS) for f in files):
            return True
    return False


def corpus_size(path: str) -> int:
    """目录下所有可入库文档的字节数（只算 DOC_EXTS，不看别的杂物）"""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(DOC_EXTS):
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total


_ROOT_WARNED = False


def discover_corpora(docs_dir: str = DOCS_DIR, include_root: bool = False) -> dict:
    """docs/ 下每个子目录算一个独立知识库。

    为什么要分开：学校制度和公司制度混在一个索引里，问学校的问题可能召回公司条文，
    评测指标会被「跨领域干扰」污染，也无法分别看两套文档的效果。

    根目录散文件默认**不算**知识库：以前会自动变成 default，界面上凭空多出一项，
    而且和 docs/<sub>/ 里的同名文档重复入库。现在只有 docs/ 下一个子目录都没有时，
    才回退到 default，保证「所有文档平铺在 docs/」的老用法仍然能跑。
    返回 {语料名: 目录}，只包含真正有文档的目录。
    """
    global _ROOT_WARNED
    corpora = {}
    if not os.path.isdir(docs_dir):
        return corpora
    for name in sorted(os.listdir(docs_dir)):
        path = os.path.join(docs_dir, name)
        if os.path.isdir(path) and not name.startswith(".") and _has_docs(path):
            corpora[name] = path

    root_has_docs = _has_docs(docs_dir, direct_only=True)
    if root_has_docs and (include_root or not corpora):
        corpora[DEFAULT_CORPUS] = docs_dir
    elif root_has_docs and not _ROOT_WARNED:
        _ROOT_WARNED = True
        print("[rag_core] docs/ 根目录下的散文件不会被收录，请放进子目录（如 docs/school/）")
    return corpora


def list_corpora() -> list:
    return list(discover_corpora().keys())


def default_corpus(corpora: dict | None = None) -> str | None:
    """默认知识库：文档总量最大的那个。

    以前 resolve_corpus(None) 用 next(iter(corpora))，而 dict 是按子目录名字母序的，
    实际选中的是 company，界面打开的却是 school —— 两边默认值不一致，
    `get_status(None)` / `init_retriever(None)` 会悄悄操作到另一个库上。
    """
    corpora = discover_corpora() if corpora is None else corpora
    if not corpora:
        return None
    return max(corpora, key=lambda n: (corpus_size(corpora[n]), n))


def resolve_corpus(corpus: str | None = None):
    """语料名 → (名字, 目录, 是否递归)
    corpus=None 时用 default_corpus()（文档量最大的库）。
    """
    corpora = discover_corpora()
    if not corpora:
        return (corpus or DEFAULT_CORPUS), DOCS_DIR, False
    if corpus is None:
        corpus = default_corpus(corpora) or next(iter(corpora))
    if corpus not in corpora:
        raise KeyError(f"未知知识库「{corpus}」，可选：{', '.join(corpora)}")
    return corpus, corpora[corpus], corpus != DEFAULT_CORPUS


def collection_name(corpus: str) -> str:
    """语料名 → Chroma collection 名（只允许字母/数字/_/-，3~63 字符）

    必须带 hash 后缀：中文名经 `re.sub` 全被替换成下划线，
    「学校制度」和「公司制度」都会变成 `kb`，两个库撞进同一个 collection，
    互相覆盖。hash 保证不同语料名一定落到不同集合上。
    """
    raw = str(corpus)
    clean = re.sub(r"[^0-9A-Za-z_-]+", "_", raw).strip("_-") or "kb"
    if not clean[0].isalpha():
        clean = f"kb_{clean}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
    return f"{clean[:44]}_{digest}"


# ------------------------- 文档加载与分块 -------------------------
def load_docs(docs_dir: str = DOCS_DIR, recursive: bool = True):
    """加载目录下的 PDF / MD / TXT；recursive=False 只看当前层（default 语料用）"""
    if not os.path.isdir(docs_dir):
        return []
    pattern = "**/*" if recursive else "*"
    pdf = DirectoryLoader(docs_dir, glob=f"{pattern}.pdf", loader_cls=PyPDFLoader).load()
    md = DirectoryLoader(docs_dir, glob=f"{pattern}.md", loader_cls=TextLoader,
                         loader_kwargs={"encoding": "utf-8"}).load()
    txt = DirectoryLoader(docs_dir, glob=f"{pattern}.txt", loader_cls=TextLoader,
                          loader_kwargs={"encoding": "utf-8"}).load()
    docs = []
    for d in pdf + md + txt:
        if (d.page_content or "").strip():
            docs.append(d)
    return docs


def adaptive_params(length: int):
    for limit, cs, ov in ADAPTIVE_TIERS:
        if length <= limit:
            return cs, ov
    return ADAPTIVE_TIERS[-1][1], ADAPTIVE_TIERS[-1][2]


def make_chunk_id(source: str, index: int, content: str) -> str:
    """稳定 ID：来源 + 块序号 + 内容哈希。
    用途有两个：
      1) 融合时按块去重（不再用 page_content[:100]，避免不同块被误合并）
      2) 跨进程对齐 BM25 与向量库里的同一块
    """
    raw = f"{source}|{index}|{content}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def chunk_key(doc) -> str:
    """取块的唯一键；老索引没有 chunk_id 时按内容哈希兜底"""
    meta = doc.metadata or {}
    cid = meta.get("chunk_id")
    if cid:
        return str(cid)
    src = str(meta.get("source", ""))
    page = meta.get("page", "")
    return hashlib.sha1(f"{src}|{page}|{doc.page_content}".encode("utf-8")).hexdigest()[:16]


# ------------------------- 中文分词（BM25 用） -------------------------
_CN_RUN = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_%.\-]+")


def cn_tokenize(text: str) -> list:
    """中文 BM25 分词：连续汉字切成二元组，英文/数字按整词保留。

    为什么需要：langchain 默认的 default_preprocessing_func 按 \\W+ 切词，
    一整段中文会被切成一个 token，BM25 退化成「整段完全匹配」，
    实测把「宿舍管理红线」的问题排到了一篇完全无关的网络管理办法上。
    二元组不需要装 jieba，对制度条文这类短句召回已经够用。
    """
    tokens = []
    for seg in _TOKEN.findall((text or "").lower()):
        if _CN_RUN.fullmatch(seg):
            if len(seg) == 1:
                tokens.append(seg)
            else:
                tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            tokens.append(seg)
    return tokens


def split_docs(docs, cfg: dict):
    """按配置分块，并给每块写入 chunk_index / chunk_size_used / chunk_id"""
    cfg = normalize(cfg)
    chunks = []
    for doc in docs:
        if cfg["adaptive"]:
            cs, ov = adaptive_params(len(doc.page_content))
        else:
            cs, ov = cfg["chunk_size"], cfg["chunk_overlap"]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs, chunk_overlap=ov, separators=SEPARATORS)
        parts = splitter.split_documents([doc])
        source = str((doc.metadata or {}).get("source", ""))
        for i, part in enumerate(parts):
            if not (part.page_content or "").strip():
                continue
            part.metadata = dict(part.metadata or {})
            part.metadata["chunk_index"] = i
            part.metadata["chunk_size_used"] = cs
            part.metadata["chunk_id"] = make_chunk_id(source, i, part.page_content)
            chunks.append(part)
    return chunks


def load_chunks(cfg: dict | None = None, corpus: str | None = None):
    """返回 (原始文档数, chunks)；corpus=None 时按 resolve_corpus 的默认语料"""
    cfg = cfg or load_settings()
    _, path, recursive = resolve_corpus(corpus)
    docs = load_docs(path, recursive=recursive)
    return len(docs), split_docs(docs, cfg)


def estimate_chunks(cfg: dict, corpus: str | None = None) -> dict:
    """不调 embedding 的纯文本预估，返回 {total, avg, per_doc:[(name, cs, ov, n)], docs}"""
    cfg = normalize(cfg)
    _, path, recursive = resolve_corpus(corpus)
    docs = load_docs(path, recursive=recursive)
    per_doc, total, chars = [], 0, 0
    for d in docs:
        one_cfg = dict(cfg)
        if cfg["adaptive"]:
            cs, ov = adaptive_params(len(d.page_content))
            one_cfg["chunk_size"], one_cfg["chunk_overlap"] = cs, ov
            one_cfg["adaptive"] = False
        parts = split_docs([d], one_cfg)
        total += len(parts)
        chars += sum(len(p.page_content) for p in parts)
        per_doc.append((
            os.path.basename(str((d.metadata or {}).get("source", "文档"))),
            one_cfg["chunk_size"], one_cfg["chunk_overlap"], len(parts),
        ))
    return {"total": total, "avg": chars / max(1, total),
            "per_doc": per_doc, "docs": len(docs)}


# ------------------------- 检索 -------------------------
def open_vectorstore(cfg: dict | None = None, collection: str = COLLECTION,
                     corpus: str | None = None) -> Chroma:
    if corpus is not None:
        collection = collection_name(corpus)
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        collection_name=collection,
    )


class Reranker:
    """可选精排（对 RRF 粗排出来的 candidate_k 条候选再排一次）。

    四种实现：
      off    不精排，直接用 RRF（默认，最快）
      embed  向量重排：用 bge-m3 把「问题 × 候选全文」重新算余弦相似度
             （0.1~0.3s，结果稳定，不需要额外模型）
      llm    让本地 qwen 给候选打相关性分（慢，7b 约 10s+，小模型判断不稳定）
      cross  bge-reranker 交叉编码器（效果最好，但需 sentence-transformers 与模型文件）

    实测备注：本地 3b/7b 做 listwise 排序时经常照抄示例或给出与内容无关的顺序，
    所以默认不开 llm；想追求效果优先装 cross。
    """

    def __init__(self, backend: str = "off", model: str | None = None):
        self.backend = backend
        default_model = {"llm": LLM_MODEL, "cross": "BAAI/bge-reranker-v2-m3",
                         "embed": EMBED_MODEL}.get(backend, "")
        # 向量重排固定用 bge-m3，不吃配置里的 llm 模型名
        self.model_name = EMBED_MODEL if backend == "embed" else (model or default_model)
        # llm / cross 的模型对象延迟创建，避免启动就占内存
        self._impl = None

    def _llm(self):
        if self._impl is None:
            self._impl = ChatOllama(model=self.model_name, temperature=0)
        return self._impl

    def _cross_model(self):
        if self._impl is None:
            from sentence_transformers import CrossEncoder  # 延迟导入
            self._impl = CrossEncoder(self.model_name)
        return self._impl

    @property
    def label(self) -> str:
        return {
            "off": "不精排（RRF）",
            "embed": f"向量重排（{EMBED_MODEL} 余弦）",
            "llm": f"LLM 精排（{self.model_name}）",
            "cross": f"bge-reranker（{self.model_name}）",
        }.get(self.backend, self.backend)

    def rerank(self, query: str, docs, top_n: int | None = None):
        if not docs or self.backend == "off":
            return docs[:top_n] if top_n else docs
        try:
            if self.backend == "embed":
                ranked = self._rerank_embed(query, docs)
            elif self.backend == "cross":
                pairs = [(query, d.page_content[:1200]) for d in docs]
                scores = self._cross_model().predict(pairs)
                order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
                ranked = [docs[i] for i in order]
            else:
                ranked = self._rerank_llm(query, docs)
        except Exception as e:
            print(f"[rag_core] 精排失败，回退到 RRF 顺序：{e}")
            ranked = docs
        return ranked[:top_n] if top_n else ranked

    # ---- 向量重排：查询与候选全文的余弦相似度 ----
    def _rerank_embed(self, query: str, docs):
        emb = _embedding()
        qv = emb.embed_query(query)
        dvs = emb.embed_documents([d.page_content for d in docs])
        scored = []
        for i, dv in enumerate(dvs):
            scored.append((_cosine(qv, dv), i))
        scored.sort(reverse=True)
        return [docs[i] for _, i in scored]

    # ---- LLM 精排：逐条打分 ----
    def _rerank_llm(self, query: str, docs):
        lines = []
        for i, d in enumerate(docs, 1):
            name = os.path.basename(str((d.metadata or {}).get("source", "文档")))
            body = re.sub(r"\s+", " ", d.page_content.strip())[:350]
            lines.append(f"[{i}]（{name}）{body}")
        skeleton = ",".join(f'{{"id":{i},"score":0}}' for i in range(1, len(docs) + 1))
        prompt = (
            "你是检索结果精排器。对下面每个候选片段判断它能否回答用户问题，打 0-10 分"
            "（10=直接给出答案，5=部分相关，0=完全无关）。\n"
            f"必须为全部 {len(docs)} 个候选分别打分，score 用整数，不要照抄示例中的 0，不要解释。\n"
            f'只输出 JSON：{{"ranking":[{skeleton}]}}\n\n'
            f"问题：{query}\n\n候选：\n" + "\n".join(lines)
        )
        raw = self._llm().invoke(prompt).content or ""
        data = _parse_json(raw)
        items = (data or {}).get("ranking") or []
        score_map = {}
        for it in items:
            try:
                score_map[int(it["id"])] = float(it["score"])
            except Exception:
                continue
        # 分数不全或全相同（模型照抄示例），说明没认真评，退回 RRF 顺序
        if len(score_map) < len(docs) or len(set(score_map.values())) <= 1:
            return docs
        return [d for _, d in sorted(
            enumerate(docs), key=lambda p: score_map.get(p[0] + 1, 0), reverse=True)]


_EMB = None


def _embedding() -> OllamaEmbeddings:
    global _EMB
    if _EMB is None:
        _EMB = OllamaEmbeddings(model=EMBED_MODEL)
    return _EMB


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


def _parse_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


class HybridRetriever:
    """BM25（关键词）+ 向量（语义）双路召回，RRF 融合，可选精排。

    融合公式：score(d) = Σ_weight  w / (rrf_k + rank)，rrf_k 来自配置（默认 20）。
      rrf_k 越大，排名差距被抹平得越多（60 是常见默认值，但小语料里过于扁平，
      第 1 名和第 12 名几乎同分，换成 20 才能让头部真正占优）。
    去重键用 chunk_id（source + 块序号 + 内容哈希），
    不再用 page_content[:100] —— 那种写法会把开头相同、后半不同的块错误合并。
    """

    def __init__(self, bm25, vector, cfg: dict, reranker: Reranker | None = None,
                 corpus: str | None = None):
        self.bm25 = bm25
        self.vector = vector
        self.cfg = normalize(cfg)
        self.reranker = reranker
        self.corpus = corpus

    # LangChain 接口兼容
    def invoke(self, query: str):
        return [d for d, _ in self.invoke_with_meta(query)]

    def invoke_with_meta(self, query: str):
        cfg = self.cfg
        rrf_k = cfg["rrf_k"]
        bm25_docs = self.bm25.invoke(query)
        vec_docs = self.vector.invoke(query)

        pool = {}
        for rank, doc in enumerate(bm25_docs, 1):
            item = pool.setdefault(chunk_key(doc), {"doc": doc, "score": 0.0})
            item["score"] += cfg["bm25_weight"] / (rrf_k + rank)
            item["bm25_rank"] = rank
        for rank, doc in enumerate(vec_docs, 1):
            item = pool.setdefault(chunk_key(doc), {"doc": doc, "score": 0.0})
            item["score"] += cfg["vector_weight"] / (rrf_k + rank)
            item["vector_rank"] = rank

        ranked = sorted(pool.values(), key=lambda x: x["score"], reverse=True)
        # 候选窗口不能被 candidate_k 砍到比 top_k 还小，否则最终片段数不够
        window = max(cfg["candidate_k"], cfg["top_k"])
        candidates = [item["doc"] for item in ranked][:window]
        meta_by_key = {chunk_key(item["doc"]): item for item in ranked}

        if self.reranker and self.reranker.backend != "off":
            before = [chunk_key(d) for d in candidates]
            candidates = self.reranker.rerank(query, candidates)
            rerank_order = {chunk_key(d): i for i, d in enumerate(candidates, 1)}
            for key, item in meta_by_key.items():
                item["rerank_rank"] = rerank_order.get(key)

        rows = []
        for i, doc in enumerate(candidates[: cfg["top_k"]], 1):
            item = dict(meta_by_key.get(chunk_key(doc), {"doc": doc}))
            item["final_rank"] = i
            rows.append((doc, item))
        return rows


def build_retriever(cfg: dict | None = None, vs: Chroma | None = None,
                    corpus: str | None = None) -> HybridRetriever | None:
    """构建某个知识库的检索器：BM25 与向量库使用同一份切片配置，避免口径错位"""
    cfg = cfg or load_settings()
    name, _, _ = resolve_corpus(corpus)
    n_docs, chunks = load_chunks(cfg, corpus=name)
    if not chunks:
        return None
    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=cn_tokenize)
    bm25.k = cfg["candidate_k"]
    vs = vs or open_vectorstore(cfg, corpus=name)
    vector = vs.as_retriever(search_kwargs={"k": cfg["candidate_k"]})
    reranker = Reranker(cfg["rerank"], cfg["rerank_model"]) if cfg["rerank"] != "off" else None
    return HybridRetriever(bm25, vector, cfg, reranker, corpus=name)


def format_docs(docs) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        name = os.path.basename(str((d.metadata or {}).get("source", "文档")))
        out.append(f"[片段{i} · {name}]\n{d.page_content}")
    return "\n\n".join(out)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """你是一个严谨的企业助手。
请仅根据下方【上下文】回答问题；如果上下文没有相关信息，直接回复"根据提供的资料无法回答"，不要编造。

【上下文】
{context}

【问题】
{question}
"""
    )


def get_llm(model: str = LLM_MODEL, **kwargs) -> ChatOllama:
    return ChatOllama(model=model, temperature=0, **kwargs)


# ------------------------- 索引维护 -------------------------
# index_report 会被状态卡片、/config、评测反复调用；直接查 sqlite 每次都要开库。
# 以 chroma.sqlite3 的 (mtime_ns, size) 作为版本号缓存元数据，文件一变自动失效。
_INDEX_CACHE: dict = {}


def invalidate_index_cache():
    _INDEX_CACHE.clear()


def _index_meta() -> dict:
    """缓存的 sqlite 元数据：每个 collection 的 chunk 数 + 所有 segment id"""
    db = os.path.join(DB_DIR, "chroma.sqlite3")
    st = os.stat(db)
    sig = (st.st_mtime_ns, st.st_size)
    if _INDEX_CACHE.get("sig") == sig:
        return _INDEX_CACHE["meta"]

    con = sqlite3.connect(db)
    try:
        collections = list(con.execute("SELECT id, name FROM collections"))
        seg_rows = list(con.execute("SELECT id, collection FROM segments"))
        counts = dict(con.execute(
            "SELECT segment_id, COUNT(*) FROM embeddings GROUP BY segment_id"))
    finally:
        con.close()

    per_collection = {}
    for seg_id, col_id in seg_rows:
        per_collection[col_id] = per_collection.get(col_id, 0) + counts.get(seg_id, 0)

    meta = {
        "collections": [{"name": name, "chunks": per_collection.get(cid, 0)}
                        for cid, name in collections],
        "segment_ids": {seg_id for seg_id, _ in seg_rows},
    }
    _INDEX_CACHE["sig"] = sig
    _INDEX_CACHE["meta"] = meta
    return meta


def index_report(corpus: str | None = None, refresh: bool = False) -> dict:
    """报告各知识库的 chunk 数、当前使用的 collection、磁盘上的孤儿目录

    字段语义（以前 `chunks` 一个字段在两种调用下含义不同，容易误用）：
      db_file         sqlite 库文件是否存在
      corpora         有文档的知识库名列表
      active          当前语料对应的 collection 名（corpus 为 None 时是 None）
      collections     每个 collection 的名字与块数
      active_chunks   **当前语料**的块数（只在传了 corpus 时有效，否则 None）
      total_chunks    所有 collection 的块数总和
      by_corpus       每个知识库各自的块数
      orphan_dirs     磁盘上不属于任何 segment 的残留目录
    """
    db = os.path.join(DB_DIR, "chroma.sqlite3")
    report = {
        "db_file": os.path.exists(db),
        "corpora": list_corpora(),
        "active": None,
        "collections": [],
        "active_chunks": None,
        "total_chunks": 0,
        "by_corpus": {},
        "orphan_dirs": [],
    }
    if not report["db_file"]:
        return report
    if refresh:
        invalidate_index_cache()

    meta = _index_meta()
    by_name = {c["name"]: c["chunks"] for c in meta["collections"]}
    report["collections"] = meta["collections"]
    report["by_corpus"] = {name: by_name.get(collection_name(name), 0)
                           for name in report["corpora"]}
    report["total_chunks"] = sum(by_name.values())
    if corpus:
        report["active"] = collection_name(corpus)
        report["active_chunks"] = by_name.get(report["active"], 0)

    # 孤儿目录：磁盘上有、但不属于任何 segment 的目录（删集合时 Chroma 不清理它们）
    report["orphan_dirs"] = sorted(
        name for name in os.listdir(DB_DIR)
        if os.path.isdir(os.path.join(DB_DIR, name)) and name not in meta["segment_ids"]
    )
    return report


def index_signature(corpus: str | None = None) -> tuple:
    """索引的「跨进程版本号」：(sqlite mtime, sqlite 大小, 该语料 chunk 数)。

    04_app 把检索器缓存在内存里；如果另一个进程（01_build_index / 02_query /
    03_eval）重建了索引，本进程的向量句柄和 BM25 都还是旧的，提问会拿到过期结果。
    每次取检索器时比一次签名，不一致就丢掉重建。
    """
    db = os.path.join(DB_DIR, "chroma.sqlite3")
    if not os.path.exists(db):
        return (0, 0, 0)
    try:
        st = os.stat(db)
    except OSError:
        return (0, 0, 0)
    try:
        meta = _index_meta()          # 内部按 (mtime, size) 缓存，不会反复读 sqlite
        if corpus:
            target = collection_name(corpus)
            n = next((c["chunks"] for c in meta["collections"] if c["name"] == target), 0)
        else:
            n = sum(c["chunks"] for c in meta["collections"])
    except Exception:
        n = -1
    return (st.st_mtime_ns, st.st_size, n)


def drop_stale_collections(keep: set) -> list:
    """删掉不再对应任何知识库的 collection（例如改名/重构后残留的 demo）。

    通过已有 Chroma 句柄的底层 client 按名字删，避免再建一个新连接；
    删不掉的只跳过，不影响主流程。
    """
    meta = _index_meta()
    stale = [c["name"] for c in meta["collections"] if c["name"] not in keep]
    if not stale:
        return []

    dropped = []
    for name in stale:
        try:
            handle = open_vectorstore(collection=name)
            client = getattr(handle, "_client", None)
            if client is not None:
                client.delete_collection(name)
            else:
                handle.delete_collection()
            dropped.append(name)
        except Exception:
            continue
    if dropped:
        invalidate_index_cache()
    return dropped


def purge_orphan_segments() -> dict:
    """删除不在 segments 表里的 HNSW 目录。
    Chroma 的 delete_collection() 只清元数据，段目录会留在磁盘上；
    这些目录不被任何 collection 引用，删掉安全。

    返回 {"removed": [...], "failed": [...]}：
    文件被进程占用（Windows 常见）时 rmtree 会失败，这里如实记录，
    交给调用方提示用户（重启程序后再清一次即可）。
    """
    result = {"removed": [], "failed": []}
    if not os.path.exists(os.path.join(DB_DIR, "chroma.sqlite3")):
        return result
    for name in index_report(refresh=True)["orphan_dirs"]:
        path = os.path.join(DB_DIR, name)
        try:
            shutil.rmtree(path)
        except Exception:
            pass
        if os.path.exists(path):
            result["failed"].append(name)
        else:
            result["removed"].append(name)
    if result["removed"]:
        invalidate_index_cache()
    return result


def clear_all_index() -> dict:
    """删除全部知识库的 collection 与残留 HNSW 目录，**不重建**。

    与 rebuild_all 的区别：rebuild 是「删完立刻按当前参数重写」，清理只做前半段，
    把「什么时候重建」交回给用户（例如想先调好切片参数再建库）。
    返回 {"dropped", "purged", "purge_failed", "failed"}，失败项如实记录不吞掉。
    """
    result = {"dropped": [], "purged": [], "purge_failed": [], "failed": []}
    try:
        names = [c["name"] for c in _index_meta()["collections"]]
    except Exception as e:
        result["failed"].append(f"读取索引元数据失败：{e}")
        return result

    for name in names:
        try:
            open_vectorstore(collection=name).delete_collection()
            result["dropped"].append(name)
        except Exception as e:
            result["failed"].append(f"{name}: {e}")

    invalidate_index_cache()
    purged = purge_orphan_segments()
    result["purged"], result["purge_failed"] = purged["removed"], purged["failed"]
    return result


def rebuild_index(cfg: dict | None = None, corpus: str | None = None,
                  progress=None) -> dict:
    """重建单个知识库：删旧集合 → 清孤儿目录 → 重新分块入库"""
    cfg = save_settings(cfg or load_settings())
    name, _, _ = resolve_corpus(corpus)
    stats = {"corpus": name, "deleted": False, "purged": [], "purge_failed": [],
             "docs": 0, "chunks": 0}

    # 1. 删 collection（元数据）
    try:
        open_vectorstore(corpus=name).delete_collection()
        stats["deleted"] = True
    except Exception:
        pass
    invalidate_index_cache()

    # 2. 清 HNSW 残留目录
    purged = purge_orphan_segments()
    stats["purged"], stats["purge_failed"] = purged["removed"], purged["failed"]

    # 3. 重新分块 + 写入
    n_docs, chunks = load_chunks(cfg, corpus=name)
    stats["docs"], stats["chunks"] = n_docs, len(chunks)
    if not chunks:
        return stats

    vs = open_vectorstore(cfg, corpus=name)
    batch = 32
    for start in range(0, len(chunks), batch):
        vs.add_documents(chunks[start:start + batch])
        if progress:
            progress("add", name, min(start + batch, len(chunks)), len(chunks))
    invalidate_index_cache()
    return stats


def rebuild_all(cfg: dict | None = None, progress=None) -> dict:
    """重建 docs/ 下所有知识库，并清掉不对应任何语料的旧 collection"""
    cfg = save_settings(cfg or load_settings())
    names = list_corpora()
    total = {"corpora": {}, "docs": 0, "chunks": 0, "purged": [], "purge_failed": []}

    if progress:
        progress("start", "", 0, len(names))

    for name in names:
        st = rebuild_index(cfg, corpus=name, progress=progress)
        total["corpora"][name] = {"docs": st["docs"], "chunks": st["chunks"]}
        total["docs"] += st["docs"]
        total["chunks"] += st["chunks"]
        total["purged"] += st["purged"]
        total["purge_failed"] += st["purge_failed"]

    total["dropped"] = drop_stale_collections({collection_name(n) for n in names})
    invalidate_index_cache()
    return total
