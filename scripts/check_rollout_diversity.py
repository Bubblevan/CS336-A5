#!/usr/bin/env python3
"""
检查 Expert Iteration Rollout 的多样性：
对于每个 prompt 生成 4 个 response，统计其中有多少个是不同的（非全同），
并展示存在差异的具体例子，帮助判断 rollout 是否真的有多样性。
"""
import json
import os
import sys
from collections import Counter

ROLLOUT_DIR = "/root/gpufree-data/cs336/outputs/expert_iteration_v2"
ROUNDS = ["rollouts_round1.jsonl", "rollouts_round2.jsonl", "rollouts_round3.jsonl"]


def load_rollouts(filepath):
    """将 jsonl 按 prompt 分组，每个 prompt 对应一个 response 列表"""
    prompts = {}
    with open(filepath) as f:
        for line in f:
            d = json.loads(line)
            p = d["prompt"]
            r = d["response"]
            prompts.setdefault(p, []).append(r)
    return prompts


def analyze_round(filepath, label):
    prompts = load_rollouts(filepath)
    total_prompts = len(prompts)

    # ---------- 每个 prompt 的 response 数量分布 ----------
    size_counter = Counter()
    for ress in prompts.values():
        size_counter[len(ress)] += 1

    # ---------- 4 response 的 prompt 中：全同 vs 有差异 ----------
    all_identical = 0
    has_different = 0
    different_examples = []  # 存几个例子
    for p, ress in prompts.items():
        if len(ress) == 4:
            if len(set(ress)) == 1:
                all_identical += 1
            else:
                has_different += 1
                if len(different_examples) < 3:
                    different_examples.append((p, ress))

    # ---------- 对于"有差异"的 prompt，看看每对间的编辑距离 ----------
    def levenshtein(a, b, max_dist=50):
        """提前截断的编辑距离 —— 差异超过 max_dist 直接返回上限，节省计算"""
        # 统一截断到 <= 1000 字符
        a, b = a[:1000], b[:1000]
        n, m = len(a), len(b)
        if n == 0:
            return min(m, max_dist)
        if m == 0:
            return min(n, max_dist)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                tmp = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = tmp
            if min(dp) > max_dist:
                return max_dist
        return min(dp[m], max_dist)

    edit_distances = []
    for p, ress in prompts.items():
        if len(ress) == 4 and len(set(ress)) > 1:
            # 计算 pairwise 最小编辑距离
            min_ed = min(levenshtein(ress[i], ress[j])
                         for i in range(4) for j in range(i + 1, 4))
            edit_distances.append(min_ed)

    # ========== 输出报告 ==========
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  总行数:               {sum(size_counter.values())}")
    print(f"  唯一 prompt 数:        {total_prompts}")
    print(f"  每个 prompt 的 response 数分布: {dict(sorted(size_counter.items()))}")
    print()
    if has_different + all_identical > 0:
        total_4 = has_different + all_identical
        print(f"  有 4 个 response 的 prompt 总数: {total_4}")
        print(f"    ✅ 4个全部相同:      {all_identical} ({all_identical/total_4*100:.1f}%)")
        print(f"    ❌ 存在不同:         {has_different} ({has_different/total_4*100:.1f}%)")
        if edit_distances:
            avg_ed = sum(edit_distances) / len(edit_distances)
            print(f"  差异 prompt 的平均编辑距离: {avg_ed:.1f} 字符")
        print()
    else:
        print("  (没有 4 个 response 的 prompt)")
        print()

    # ---------- 展示例子 ----------
    if different_examples:
        print(f"  --- 展示 {len(different_examples)} 个存在差异的例子 ---")
        for idx, (p, ress) in enumerate(different_examples):
            print(f"\n  [例 {idx+1}] Prompt (截断): {p[:120]}...")
            for i, r in enumerate(ress):
                # 找出和其他 response 不同的标记
                mark = " *** 不同 ***" if sum(1 for rr in ress if rr == r) < 4 else ""
                r_short = r[:250].replace("\n", "\\n")
                print(f"    Response[{i}]: {r_short}{mark}")
    else:
        print("  (没有存在差异的例子)")

    print()

    return all_identical, has_different


def main():
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

    results = []
    for fname in ROUNDS:
        fpath = os.path.join(ROLLOUT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] 文件不存在: {fpath}")
            continue
        label = fname.replace("rollouts_", "Rollout ").replace(".jsonl", "")
        all_id, has_diff = analyze_round(fpath, label)
        results.append((label, all_id, has_diff))

    # ---------- 总结 ----------
    print(f"\n{'='*70}")
    print(f"  结论汇总")
    print(f"{'='*70}")
    print(f"  {'轮次':<20} {'全同(4个一样)':>15} {'有差异':>10} {'有差异占比':>12}")
    print(f"  {'-'*57}")
    for label, all_id, has_diff in results:
        total = all_id + has_diff
        pct = f"{has_diff/total*100:.1f}%" if total > 0 else "N/A"
        print(f"  {label:<20} {all_id:>15} {has_diff:>10} {pct:>12}")
    print()

    print("  💡 解读:")
    print("     - 如果全同占比接近 100%，说明 rollout 生成时 temperature 可能为 0")
    print("       或模型在给定 prompt 下确定性极强 (model collapse)。")
    print("     - 这种情况下 GRPO 的 advantage 计算会失效")
    print("       (所有 response 的 reward 相同，advantage 恒为 0)，")
    print("       模型实际上没有学习到新东西。")
    print("     - 如果验证指标有提升，可能是 reward model 本身的偏差，")
    print("       或者评估集与训练集分布不一致导致的假象。")
    print()


if __name__ == "__main__":
    main()
