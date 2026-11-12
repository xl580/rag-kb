"""
Step 3: 评测（召回 & 幻觉，两套指标分开算）

之前的问题：
  1) 幻觉题混在召回集里，用 "文档无此信息" in joined 判断 —— 检索结果里当然不会出现
     这句话，那几条必然判 ✗，命中率被无意义拉低，也无法说明模型是否真的会拒答；
  2) 所有题共用一个索引，学校题可能召回公司条文 —— 指标被跨领域干扰污染；
  3) 索引只有 4 个 chunk，Top-4 等于全量返回，Recall@4 必然 100%，没有区分度。

现在的做法：
  1) 召回评测：只问「知识库里查得到的问题」，算 Recall@K / Recall@1 / MRR
  2) 幻觉评测：只问「知识库里没有的问题」，真的调用 LLM，看它是否拒不编造
  3) 每个知识库（docs/ 下的子目录）用自己的 collection 评，互不干扰
  4) 索引 chunk 数 <= K 时明确警告指标饱和

两者都读 rag_ui_config.json，与 01/02/04 用同一份切片参数，
所以这里的结果才代表 UI 的实际行为。

用法：
    python 03_eval.py                  # 召回 + 幻觉
    python 03_eval.py --recall-only
    python 03_eval.py --halluc-only
    python 03_eval.py --rerank embed   # 临时开精排对比
"""
import argparse

import rag_core as core

# ---------------- 测试集：知识库里有答案的 ----------------
# corpus 与 docs/ 下的子目录同名，各自在自己的索引里检索
RECALL_SET = [
    # —— 晋中信息学院学生管理规定（docs/school/jzci_rules.md）——
    {"q": "宿舍管理有哪两条红线？", "k": "违章用电", "corpus": "school"},
    {"q": "损坏校园公共财物应该怎么做？", "k": "赔偿", "corpus": "school"},
    {"q": "宿舍里能不能私自留宿外来人员？", "k": "私自留宿", "corpus": "school"},
    {"q": "学生违纪处分分为哪几个等级？", "k": "留校察看", "corpus": "school"},
    {"q": "哪些情况可以从轻处分？", "k": "悔改", "corpus": "school"},
    {"q": "主动挽回损失或阻止危害发生怎么处理？", "k": "免予处分", "corpus": "school"},
    {"q": "拒不认错、多次违纪会怎么处理？", "k": "从重", "corpus": "school"},
    {"q": "考试找他人替考会受到什么处分？", "k": "开除学籍", "corpus": "school"},
    {"q": "用手机通讯设备传递答案是什么处分？", "k": "记过", "corpus": "school"},
    {"q": "两份答卷雷同率达到多少认定为作弊？", "k": "80%", "corpus": "school"},
    {"q": "利用无线耳机作弊是什么处分？", "k": "开除学籍", "corpus": "school"},
    {"q": "伙同校外人员到校一旦发生打架怎么处理？", "k": "移交公安", "corpus": "school"},
    {"q": "打架致重伤是什么处分？", "k": "开除学籍", "corpus": "school"},
    {"q": "先动手打人会怎么处理？", "k": "从重", "corpus": "school"},
    {"q": "以劝架为名偏袒一方扩大事态怎么处理？", "k": "扩大事态", "corpus": "school"},
    {"q": "结伙斗殴没有伤人是什么处分？", "k": "留校察看", "corpus": "school"},
    {"q": "学生申诉由哪个机构受理？", "k": "学生申诉处理委员会", "corpus": "school"},
    {"q": "对同一处分可以申诉几次？", "k": "一次", "corpus": "school"},
    # —— 公司员工手册（docs/company/公司制度样例.md）——
    {"q": "出差住宿一线城市标准是多少？", "k": "400", "corpus": "company"},
    {"q": "入职满 5 年有几天年假？", "k": "10 天", "corpus": "company"},
    {"q": "迟到超过 30 分钟怎么算？", "k": "旷工", "corpus": "company"},
    {"q": "代码仓库写权限怎么申请？", "k": "主管", "corpus": "company"},
    {"q": "试用期社保怎么交？", "k": "全额", "corpus": "company"},
    {"q": "报销多久能打款？", "k": "10 个工作日", "corpus": "company"},
    {"q": "客户数据导出要谁审批？", "k": "数据安全委员会", "corpus": "company"},
    {"q": "事假一年最多请多少天？", "k": "15 天", "corpus": "company"},
]

# ---------------- 测试集：知识库里没有答案的（幻觉） ----------------
HALLUCINATION_SET = [
    ("旷课多少节给予警告处分？", "school"),
    ("星级文明宿舍评比多久开展一次？", "school"),
    ("学生转专业需要满足什么条件？", "school"),
    ("学校图书馆的开放时间是几点？", "school"),
]

REFUSAL_MARKERS = (
    "无法回答", "没有相关信息", "没有提供", "未提及", "没有提到",
    "资料中没有", "资料未涉及", "无法确定", "不能确定", "没有说明",
)


def evaluate_recall(retrievers, k, chunks_by_corpus):
    print(f"\n{'=' * 62}\n一、召回评测（每个知识库各自检索）\n{'=' * 62}")

    hit, hit1, rr_sum = 0, 0, 0.0
    by_corpus, misses = {}, []

    for item in RECALL_SET:
        corpus = item["corpus"]
        retriever = retrievers.get(corpus)
        if retriever is None:
            print(f"  ✗ [跳过] {item['q']}（知识库 {corpus} 不存在）")
            continue
        rows = retriever.invoke_with_meta(item["q"])
        contents = [d.page_content for d, _ in rows[:k]]
        rank = next((i + 1 for i, c in enumerate(contents) if item["k"] in c), None)
        ok = rank is not None
        hit += int(ok)
        hit1 += int(rank == 1)
        rr_sum += 1.0 / rank if rank else 0.0

        group = by_corpus.setdefault(corpus, {"n": 0, "hit": 0, "hit1": 0, "rr": 0.0})
        group["n"] += 1
        group["hit"] += int(ok)
        group["hit1"] += int(rank == 1)
        group["rr"] += 1.0 / rank if rank else 0.0
        if not ok:
            misses.append((item, len(contents)))

        flag = "✓" if ok else "✗"
        where = f"Top{rank}" if rank else "未命中"
        print(f"  {flag} [{where:>5}] {item['q']}  (期望含「{item['k']}」)")

    n = sum(g["n"] for g in by_corpus.values()) or 1
    print(f"\n  召回命中率 Recall@{k}: {hit}/{n} = {hit / n * 100:.1f}%")
    print(f"  Recall@1（最相关片段排第一）: {hit1}/{n} = {hit1 / n * 100:.1f}%")
    print(f"  MRR@{k}: {rr_sum / n:.3f}")

    for corpus, g in by_corpus.items():
        total = chunks_by_corpus.get(corpus, 0)
        sat = "（chunk 数 ≤ K，指标饱和）" if total <= k else ""
        print(f"    · {corpus:<8} {g['hit']}/{g['n']} = {g['hit'] / g['n'] * 100:.1f}%"
              f" · Recall@1 {g['hit1']}/{g['n']} · MRR {g['rr'] / g['n']:.3f}"
              f" · 索引 {total} chunk{sat}")

    if misses:
        print("\n  未命中明细：")
        for item, got in misses:
            print(f"    - [{item['corpus']}] {item['q']} → 只取到 {got} 个片段，"
                  f"均不含「{item['k']}」")

    return {"recall": hit / n, "recall1": hit1 / n, "mrr": rr_sum / n, "n": n}


def evaluate_hallucination(retrievers, llm, prompt, k):
    print(f"\n{'=' * 62}\n二、幻觉评测（知识库里查不到的问题，应当拒答）\n{'=' * 62}")
    refused = 0
    failures = []

    for q, corpus in HALLUCINATION_SET:
        retriever = retrievers.get(corpus)
        if retriever is None:
            print(f"  - 跳过 {q}（知识库 {corpus} 不存在）")
            continue
        rows = retriever.invoke_with_meta(q)
        docs = [d for d, _ in rows[:k]]
        messages = prompt.format_messages(context=core.format_docs(docs), question=q)
        ans = (llm.invoke(messages).content or "").strip()
        ok = any(m in ans for m in REFUSAL_MARKERS)
        refused += int(ok)
        if not ok:
            failures.append((q, ans))
        flag = "✓" if ok else "✗"
        print(f"  {flag} [{corpus}] {q}")
        print(f"      → {ans[:110].replace(chr(10), ' ')}")

    n = len(HALLUCINATION_SET)
    print(f"\n  正确拒答率: {refused}/{n} = {refused / n * 100:.1f}%")
    if failures:
        print("  编造风险题（未拒答）：")
        for q, ans in failures:
            print(f"    - {q} → {ans[:90]}")

    return {"refusal": refused / n, "n": n}


def main():
    parser = argparse.ArgumentParser(description="RAG 召回 / 幻觉评测")
    parser.add_argument("--k", type=int, default=None, help="取前 K 个片段（默认读配置）")
    parser.add_argument("--rerank", choices=["off", "embed", "llm", "cross"], help="临时覆盖精排方式")
    parser.add_argument("--recall-only", action="store_true")
    parser.add_argument("--halluc-only", action="store_true")
    args = parser.parse_args()

    cfg = core.load_settings()
    if args.k:
        cfg["top_k"] = args.k
    if args.rerank:
        cfg["rerank"] = args.rerank
    cfg = core.normalize(cfg)
    k = cfg["top_k"]

    if not core.index_report()["db_file"]:
        raise SystemExit("未找到 vectorstore/，请先运行 python 01_build_index.py")

    report = core.index_report()
    print(f"知识库：{', '.join(report['corpora']) or '（无）'}")
    print(f"索引：" + " · ".join(
        f"{name} {report['by_corpus'].get(name, 0)} chunk" for name in report["corpora"]))
    print(f"切片：{core.describe_chunking(cfg)}"
          + ("" if cfg["adaptive"] else f"（overlap={cfg['chunk_overlap']}）"))
    print(f"检索：BM25 {cfg['bm25_weight']} + 向量 {cfg['vector_weight']}，"
          f"候选各 {cfg['candidate_k']} 条 → RRF(k={cfg['rrf_k']}) → Top {k}")
    print(f"精排：{core.rerank_label(cfg['rerank'])}")

    retrievers = {}
    for name in report["corpora"]:
        try:
            r = core.build_retriever(cfg, corpus=name)
        except Exception as e:
            print(f"  ✗ 知识库 {name} 检索器构建失败：{e}")
            r = None
        if r is not None:
            retrievers[name] = r
    if not retrievers:
        raise SystemExit("没有任何可用知识库")

    if not args.halluc_only:
        evaluate_recall(retrievers, k, report["by_corpus"])

    if not args.recall_only:
        evaluate_hallucination(retrievers, core.get_llm(), core.build_prompt(), k)

    print(f"\n{'=' * 62}")
    print("提示：召回看 Recall/MRR，幻觉看拒答率，两个指标不能混着算；")
    print("      Recall@K 只有在「K < chunk 总数」时才有区分度。")


if __name__ == "__main__":
    main()
