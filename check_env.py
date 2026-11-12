"""
check_env.py —— 一条命令确认运行环境（依赖 / Ollama / 模型 / 索引）

用法：
    .venv\\Scripts\\python.exe check_env.py

以前的问题是：.venv 里缺 gradio、rank_bm25，Web 界面只能用系统 Python 跑，
两个解释器的依赖还不一样。现在 .venv 是唯一环境，跑之前用它自检一遍。
"""
import importlib
import json
import os
import sys
import urllib.request

REQUIRED = [
    "langchain_core", "langchain_community", "langchain_chroma", "langchain_ollama",
    "langchain_text_splitters", "chromadb", "rank_bm25", "pypdf", "gradio",
]
OPTIONAL = {
    "sentence_transformers": "启用 bge-reranker 交叉编码器精排需要",
}
NEED_MODELS = ["bge-m3", "qwen2.5:7b"]
OLLAMA = "http://127.0.0.1:11434"


def main() -> int:
    ok = True
    print(f"Python: {sys.version.split()[0]}")
    print(f"解释器: {sys.executable}")
    if ".venv" not in sys.executable.lower():
        print("提示: 当前不是项目 .venv 里的解释器，建议用 .venv\\Scripts\\python.exe\n")
    else:
        print()

    print("依赖检查：")
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            print(f"  OK   {mod:<26} {ver}")
        except Exception as e:
            ok = False
            print(f"  MISS {mod:<26} {e}")
    for mod, why in OPTIONAL.items():
        try:
            importlib.import_module(mod)
            print(f"  OK   {mod:<26} （可选）")
        except Exception:
            print(f"  --   {mod:<26} 未安装（可选，{why}）")

    print("\nOllama 服务：")
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as r:
            tags = json.load(r)
        names = {m["name"] for m in tags.get("models", [])}
        print(f"  OK   已连接 {OLLAMA}")
        for need in NEED_MODELS:
            hit = any(n == need or n.startswith(need + ":") for n in names)
            print(f"  {'OK  ' if hit else 'MISS'} 模型 {need}")
            ok = ok and hit
    except Exception as e:
        ok = False
        print(f"  MISS 连不上 {OLLAMA}（先运行 ollama serve）：{e}")

    print("\n索引状态：")
    try:
        import rag_core as core
        rep = core.index_report()
        print(f"  知识库：{', '.join(rep['corpora']) or '（docs/ 下没有文档）'}")
        if not rep["db_file"]:
            print("  -- 还没有 vectorstore/，先跑 01_build_index.py")
        else:
            for name in rep["corpora"]:
                print(f"  · {name:<10} {rep['by_corpus'].get(name, 0)} 个 chunk"
                      f"（collection={core.collection_name(name)}）")
            print(f"  共 {rep['total_chunks']} 个 chunk · {len(rep['collections'])} 个集合")
            if rep["orphan_dirs"]:
                print(f"  ! 有 {len(rep['orphan_dirs'])} 个残留目录（启动 04_app.py 会自动清理）")
        cfg = core.load_settings()
        print(f"  切片 {core.describe_chunking(cfg)} · RRF k={cfg['rrf_k']} · "
              f"精排 {core.rerank_label(cfg['rerank'])}")
    except Exception as e:
        print(f"  读取索引状态失败：{e}")

    print("\n" + ("环境可用 ✅" if ok else "环境有问题 ❌，按上面的 MISS 修"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
