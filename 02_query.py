"""
Step 2: 混合检索 + LLM 生成（命令行版）

分块参数与 UI 共用 rag_ui_config.json：
  - 在 04_app.py 里改成 400/60 重建，这里自动跟着用 400/60，
    不会再出现「CLI 的 BM25 用 800/120、向量库是 400/60」的错位。

用法：
    python 02_query.py
    python 02_query.py --corpus school   # 指定知识库（默认取 docs/ 下的第一个）
    python 02_query.py --rerank llm      # 临时开精排
    python 02_query.py --top-k 6
交互命令：/config 看当前参数，/quit 退出
"""
import argparse
import os

import rag_core as core


def print_config(cfg, corpus=None):
    if corpus:
        print(f"  知识库：{corpus}（collection={core.collection_name(corpus)}）")
    print(f"  切片：{core.describe_chunking(cfg)}"
          + ("" if cfg["adaptive"] else f"（overlap={cfg['chunk_overlap']}）"))
    print(f"  粗排：BM25 {cfg['bm25_weight']} + 向量 {cfg['vector_weight']}，"
          f"各取 {cfg['candidate_k']} 条 → RRF(k={cfg['rrf_k']}) → Top {cfg['top_k']}")
    print(f"  精排：{core.rerank_label(cfg['rerank'])}"
          + (f"（{cfg['rerank_model']}）" if cfg["rerank"] in ("llm", "cross") else ""))


def main():
    parser = argparse.ArgumentParser(description="RAG 命令行问答")
    parser.add_argument("--top-k", type=int, help="最终使用片段数")
    parser.add_argument("--rerank", choices=["off", "embed", "llm", "cross"], help="精排方式")
    parser.add_argument("--corpus", help="知识库名（docs/ 下的子目录名）")
    args = parser.parse_args()

    cfg = core.load_settings()
    if args.top_k:
        cfg["top_k"] = args.top_k
    if args.rerank:
        cfg["rerank"] = args.rerank
    cfg = core.normalize(cfg)

    if not os.path.exists(os.path.join(core.DB_DIR, "chroma.sqlite3")):
        raise SystemExit("未找到 vectorstore/，请先运行 python 01_build_index.py")

    name, _, _ = core.resolve_corpus(args.corpus)
    print("加载索引中…")
    retriever = core.build_retriever(cfg, corpus=name)
    if retriever is None:
        raise SystemExit(f"知识库 {name} 下没有可用文档，请先放几份进去再建库")

    llm = core.get_llm()
    prompt = core.build_prompt()
    print_config(cfg, name)
    print("\n输入问题（/quit 退出，/config 看配置）：")

    while True:
        q = input("\n你> ").strip()
        if q.lower() in ("/quit", "quit", "exit", "q"):
            break
        if q == "/config":
            print_config(core.load_settings())
            continue
        if not q:
            continue

        rows = retriever.invoke_with_meta(q)
        if not rows:
            print("\n助手> 没有检索到任何片段。")
            continue

        docs = [d for d, _ in rows]
        print("\n[召回] " + " | ".join(
            f"#{m['final_rank']} {os.path.basename(str(d.metadata.get('source','?')))}"
            f"(BM25:{m.get('bm25_rank','-')}/向量:{m.get('vector_rank','-')})"
            for d, m in rows))

        messages = prompt.format_messages(context=core.format_docs(docs), question=q)
        print("\n助手> ", end="", flush=True)
        for chunk in llm.stream(messages):
            print(chunk.content or "", end="", flush=True)
        print()


if __name__ == "__main__":
    main()
