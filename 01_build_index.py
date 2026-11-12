"""
Step 1: 构建向量索引（支持多知识库）

docs/ 下每个子目录 = 一个独立知识库（如 docs/school/、docs/company/），
各自建一个 Chroma collection。分开建库是为了避免不同领域的条文混在一个索引里互相干扰。
根目录散落的文件不再自动算一个 default 库（只有 docs/ 下一个子目录都没有时才回退到 default）。

切片参数不写死：默认读 rag_ui_config.json（与 02/03/04 共用同一份配置），
命令行可临时覆盖，覆盖后写回配置，保证四个入口口径一致。

用法：
    python 01_build_index.py                          # 重建全部知识库
    python 01_build_index.py --corpus school          # 只重建 school
    python 01_build_index.py --chunk-size 400 --overlap 60
    python 01_build_index.py --adaptive               # 按文档长度自适应
运行前确保：Ollama 已启动，且已 pull qwen2.5:7b 与 bge-m3。
"""
import argparse

import rag_core as core


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
    parser.add_argument("--chunk-size", type=int, help="分块大小（写回配置）")
    parser.add_argument("--overlap", type=int, help="重叠长度（写回配置）")
    parser.add_argument("--adaptive", action="store_true", help="自适应分块")
    parser.add_argument("--corpus", help="只重建某个知识库（默认全部）")
    args = parser.parse_args()

    cfg = core.load_settings()
    if args.chunk_size is not None:
        cfg["chunk_size"] = args.chunk_size
    if args.overlap is not None:
        cfg["chunk_overlap"] = args.overlap
    if args.adaptive:
        cfg["adaptive"] = True
    cfg = core.normalize(cfg)

    corpora = core.list_corpora()
    if not corpora:
        raise SystemExit("docs/ 下没有文档，先放几份 .md/.pdf/.txt 进去")
    print(f"知识库：{', '.join(corpora)}")
    print(f"[0/3] 切片方式：{core.describe_chunking(cfg)}"
          + ("（按每篇长度自动定参）" if cfg["adaptive"] else ""))

    def progress(stage, name, done, total):
        if stage == "add":
            print(f"      {name} 向量化 {done}/{total}")

    if args.corpus:
        stats = core.rebuild_index(cfg, corpus=args.corpus, progress=progress)
        detail = {stats["corpus"]: {"docs": stats["docs"], "chunks": stats["chunks"]}}
    else:
        stats = core.rebuild_all(cfg, progress=progress)
        detail = stats["corpora"]

    print(f"[1/3] 文档 {stats['docs']} 篇")
    print(f"[2/3] 分块完成，共 {stats['chunks']} 个 chunk")
    for name, info in detail.items():
        print(f"      · {name:<10} {info['docs']} 篇 → {info['chunks']} 个 chunk"
              f"（collection={core.collection_name(name)}）")
    print("[3/3] 向量索引已写入 ./vectorstore")
    if stats.get("purged"):
        print(f"      顺带清理了 {len(stats['purged'])} 个无用索引目录")
    if stats.get("dropped"):
        print(f"      删除过期 collection：{', '.join(stats['dropped'])}")
    print("完成。下一步运行 python 02_query.py 提问。")


if __name__ == "__main__":
    main()
