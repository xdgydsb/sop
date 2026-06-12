"""
动作分割 — 子序列DTW匹配 + DP最优分段
用yuanzi模板匹配视频中的动作段，保留模板内部时序信息。

方法:
1. 子序列DTW: 对每个step模板，计算在视频每个位置结束的匹配代价
   - 保留模板帧间顺序 (i→i+1匹配j→j+1)
   - 开起点: 模板可从视频任意帧开始匹配
2. 代价转相似度 + 平滑 → 5条相似度曲线
3. DP最优分段: 找S1→S5的最优非重叠分割
4. WR视频: 逐帧最佳匹配，不强制顺序

数据:
- yuanzi/1-5.avi: 单动作模板
- ok/*.avi (182): 正确SOP → 强制S1→S5顺序
- wr/*.avi (70): 错误SOP → 不强制顺序
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple, Optional
from scipy.ndimage import gaussian_filter1d

# 全局缓存
yuanzi_features_cache = {}
template_lens = {}

# 特征维度选择
# 使用手部运动(84) + 物体存在(5) + 手物交互(20) + 盒子状态(2) = 111
KEY_DIMS = list(range(84)) + list(range(84, 89)) + list(range(89, 109)) + list(range(126, 128))


def subsequence_dtw_cost(template: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    """
    子序列DTW: 模板匹配视频子段，保留模板内部时序。

    template: (M, D) — 模板帧序列
    sequence: (N, D) — 视频帧序列

    Returns:
        costs: (N,) — 以每帧结束的完整模板匹配代价 (越低越好)
        start_frames: (N,) — 对应的起始帧
    """
    M, N = len(template), len(sequence)
    costs = np.full(N, np.inf)
    start_frames = np.zeros(N, dtype=np.int64)

    if M == 0 or N == 0 or M > N:
        return costs, start_frames

    # L2归一化
    t_norm = template / (np.linalg.norm(template, axis=1, keepdims=True) + 1e-8)
    s_norm = sequence / (np.linalg.norm(sequence, axis=1, keepdims=True) + 1e-8)

    # DTW累积代价矩阵 — 只保留两行
    prev_D = np.zeros(N + 1)  # D[0, :] = 0 (开起点)
    prev_start = np.arange(N + 1, dtype=np.int64)  # 每列对应的起始帧

    for i in range(1, M + 1):
        curr_D = np.full(N + 1, np.inf)
        curr_start = np.zeros(N + 1, dtype=np.int64)

        for j in range(1, N + 1):
            # 余弦距离 [0, 2]
            dist = 1.0 - np.dot(t_norm[i-1], s_norm[j-1])

            # 三个方向
            d_diag = prev_D[j-1] + dist
            d_up = prev_D[j] + dist     # 跳过视频帧(模板拉伸)
            d_left = curr_D[j-1] + dist  # 跳过模板帧(视频拉伸)

            if d_diag <= d_up and d_diag <= d_left:
                curr_D[j] = d_diag
                curr_start[j] = prev_start[j-1]
            elif d_up <= d_left:
                curr_D[j] = d_up
                curr_start[j] = prev_start[j]
            else:
                curr_D[j] = d_left
                curr_start[j] = curr_start[j-1]

        prev_D = curr_D
        prev_start = curr_start

    # 匹配完整模板的代价 (归一化)
    costs = prev_D[1:] / M
    start_frames = prev_start[1:]
    start_frames = np.maximum(0, start_frames - 1)  # 转为0-based

    return costs, start_frames


def compute_dtw_similarities(X: np.ndarray) -> Dict[int, np.ndarray]:
    """
    对每个step计算DTW相似度曲线。
    sim = 1.0 / (1.0 + cost) → 范围(0, 1], 代价越低相似度越高。
    """
    sim_curves = {}
    for step in range(1, 6):
        template = yuanzi_features_cache[step][:, KEY_DIMS].astype(np.float64)
        sequence = X[:, KEY_DIMS].astype(np.float64)

        costs, starts = subsequence_dtw_cost(template, sequence)

        # 代价→相似度
        # 用中位数作为基准
        finite_costs = costs[np.isfinite(costs)]
        if len(finite_costs) > 0:
            median_cost = np.median(finite_costs)
        else:
            median_cost = 1.0

        sim = 1.0 / (1.0 + costs / max(median_cost, 0.01))

        # 平滑
        sim = gaussian_filter1d(sim.astype(np.float64), sigma=2.0)

        sim_curves[step] = sim

    # 归一化到[0, 1]
    for step in range(1, 6):
        c = sim_curves[step]
        r = c.max() - c.min()
        if r > 1e-8:
            sim_curves[step] = (c - c.min()) / r
        else:
            sim_curves[step] = np.ones_like(c) * 0.5

    return sim_curves


def dp_segment(sim_curves: Dict[int, np.ndarray], n_frames: int) -> List[Dict]:
    """
    DP最优分段: 找S1→S5的最优非重叠分割。
    时长先验: 基于yuanzi模板长度比例。
    """
    N = n_frames
    K = 5
    INF_NEG = -1e10

    # 时长先验比例
    total_tpl = sum(template_lens.values())
    prior_ratios = np.array([template_lens[s] / total_tpl for s in range(1, 6)])

    min_seg = 8
    step = max(1, N // 150)  # 粗搜索步长

    dp = np.full((K + 1, N), INF_NEG)
    bt = np.zeros((K + 1, N), dtype=np.int64)
    dp[0, :] = 0

    for s in range(1, K + 1):
        sim = sim_curves[s]
        prior_dur = prior_ratios[s-1] * N

        for end in range(s * min_seg, N):
            best_score = INF_NEG
            best_start = 0
            start_min = max((s-1) * min_seg, end - int(prior_dur * 2.5))
            start_max = end - min_seg + 1
            if start_min > start_max:
                continue

            # 粗搜
            for start in range(start_min, start_max, step):
                prev = dp[s-1, start-1] if start > 0 else (0 if s == 1 else INF_NEG)
                if prev <= INF_NEG + 1:
                    continue
                seg_sim = np.mean(sim[start:end+1])
                seg_dur = end - start + 1
                dur_pen = min(((seg_dur - prior_dur) / prior_dur) ** 2, 1.0) if prior_dur > 0 else 0
                score = prev + seg_sim - 0.05 * dur_pen
                if score > best_score:
                    best_score = score
                    best_start = start

            # 在best_start附近精搜
            fine_start = max(start_min, best_start - step)
            fine_end = min(start_max, best_start + step)
            for start in range(fine_start, fine_end + 1):
                prev = dp[s-1, start-1] if start > 0 else (0 if s == 1 else INF_NEG)
                if prev <= INF_NEG + 1:
                    continue
                seg_sim = np.mean(sim[start:end+1])
                seg_dur = end - start + 1
                dur_pen = min(((seg_dur - prior_dur) / prior_dur) ** 2, 1.0) if prior_dur > 0 else 0
                score = prev + seg_sim - 0.05 * dur_pen
                if score > best_score:
                    best_score = score
                    best_start = start

            dp[s, end] = best_score
            bt[s, end] = best_start

    # 找最佳终点
    best_end = K * min_seg
    best_total = INF_NEG
    for end in range(K * min_seg, N):
        if dp[K, end] > best_total:
            best_total = dp[K, end]
            best_end = end

    # 回溯
    ends = [0] * K
    starts = [0] * K
    cur_end = best_end
    for s in range(K, 0, -1):
        ends[s-1] = cur_end
        starts[s-1] = int(bt[s, cur_end])
        cur_end = starts[s-1] - 1

    # 构建结果
    segments = []
    for step in range(1, K + 1):
        s, e = max(0, starts[step-1]), min(N-1, ends[step-1])
        if e < s:
            e = s
        sim = sim_curves[step]
        peak = s + int(np.argmax(sim[s:e+1])) if e >= s else s
        conf = float(np.mean(sim[s:e+1]))

        segments.append({
            "step": step,
            "start": int(s),
            "end": int(e),
            "peak_frame": peak,
            "conf": conf,
            "duration": int(e - s + 1),
        })

    return segments


def segment_wr_video(sim_curves: Dict[int, np.ndarray],
                     n_frames: int) -> List[Dict]:
    """WR视频: 逐帧选最佳匹配步骤，不强制顺序"""
    all_sims = np.column_stack([sim_curves[s] for s in range(1, 6)])
    best_step = np.argmax(all_sims, axis=1) + 1

    # 众数平滑 (窗口21帧)
    window = 21
    smoothed = np.zeros(n_frames, dtype=np.int64)
    for i in range(n_frames):
        w_start = max(0, i - window // 2)
        w_end = min(n_frames, i + window // 2 + 1)
        vals, counts = np.unique(best_step[w_start:w_end], return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]

    # 构建连续段
    segments = []
    cur_step = smoothed[0]
    cur_start = 0
    for i in range(1, n_frames):
        if smoothed[i] != cur_step:
            if i - cur_start >= 5:
                sim = sim_curves[cur_step][cur_start:i]
                peak = cur_start + int(np.argmax(sim))
                segments.append({
                    "step": int(cur_step),
                    "start": cur_start,
                    "end": i - 1,
                    "peak_frame": peak,
                    "conf": float(np.mean(sim)),
                    "duration": i - cur_start,
                })
            cur_step = smoothed[i]
            cur_start = i
    if n_frames - cur_start >= 5:
        sim = sim_curves[cur_step][cur_start:]
        peak = cur_start + int(np.argmax(sim))
        segments.append({
            "step": int(cur_step),
            "start": cur_start,
            "end": n_frames - 1,
            "peak_frame": peak,
            "conf": float(np.mean(sim)),
            "duration": n_frames - cur_start,
        })

    return segments


def segments_to_labels(segments: List[dict], n_frames: int) -> np.ndarray:
    labels = np.zeros(n_frames, dtype=np.int64)
    for seg in segments:
        s = max(0, seg["start"])
        e = min(n_frames - 1, seg["end"])
        if s <= e:
            labels[s:e+1] = seg["step"]
    last_end = max((s["end"] for s in segments if s["end"] >= 0), default=-1)
    if last_end > 0 and last_end < n_frames - 1:
        labels[last_end+1:] = 6
    return labels


def load_lists(data_dir: str) -> Tuple[List[str], List[str]]:
    data_dir = Path(data_dir)
    ok = sorted([v.stem for v in (data_dir / "ok").glob("*.avi")])
    wr = sorted([v.stem for v in (data_dir / "wr").glob("*.avi")])
    return ok, wr


def compute_segmentation_all(features_dir: str, data_dir: str,
                             output_dir: str, ok_limit: int = None):
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("子序列DTW + DP最优分段")
    print("="*60)

    global yuanzi_features_cache, template_lens
    for i in range(1, 6):
        X_path = features_dir / f"{i}_X.npy"
        if X_path.exists():
            yuanzi_features_cache[i] = np.load(X_path).astype(np.float32)
            template_lens[i] = len(yuanzi_features_cache[i])
            print(f"  yuanzi/{i}: {yuanzi_features_cache[i].shape} (len={template_lens[i]})")

    if len(yuanzi_features_cache) < 5:
        print(f"  ERROR: Only {len(yuanzi_features_cache)}/5 yuanzi features!")
        return

    ok_names, wr_names = load_lists(data_dir)
    print(f"\n数据: OK={len(ok_names)}, WR={len(wr_names)}")

    # --- OK视频 ---
    ok_list = []
    for name in ok_names:
        p = features_dir / f"{name}_X.npy"
        if p.exists():
            ok_list.append((name, p))
    if ok_limit:
        ok_list = ok_list[:ok_limit]

    print(f"\n[1/2] 处理 {len(ok_list)} 个OK视频...")
    ok_count = 0
    for name, X_path in tqdm(ok_list, desc="Seg OK"):
        X = np.load(X_path).astype(np.float32)
        sim_curves = compute_dtw_similarities(X)
        segments = dp_segment(sim_curves, len(X))
        labels = segments_to_labels(segments, len(X))
        np.save(output_dir / f"{name}_y_seg.npy", labels)
        ok_count += 1

        if ok_count <= 5:
            print(f"\n  [{name}] ({len(X)} frames)")
            for seg in segments:
                print(f"    S{seg['step']}: [{seg['start']:>4d}-{seg['end']:>4d}] "
                      f"dur={seg['duration']:>3d}  peak={seg['peak_frame']:>4d}  "
                      f"conf={seg['conf']:.3f}")

    # --- WR视频 ---
    wr_list = []
    for name in wr_names:
        p = features_dir / f"{name}_X.npy"
        if p.exists():
            wr_list.append((name, p))

    print(f"\n[2/2] 处理 {len(wr_list)} 个WR视频 (不强制顺序)...")
    wr_count = 0
    for name, X_path in tqdm(wr_list, desc="Seg WR"):
        X = np.load(X_path).astype(np.float32)
        sim_curves = compute_dtw_similarities(X)
        segments = segment_wr_video(sim_curves, len(X))
        labels = segments_to_labels(segments, len(X))
        np.save(output_dir / f"{name}_y_seg.npy", labels)
        wr_count += 1

        if wr_count <= 3:
            print(f"\n  [{name}] ({len(X)} frames)")
            for seg in segments[:10]:
                print(f"    S{seg['step']}: [{seg['start']:>4d}-{seg['end']:>4d}] "
                      f"dur={seg['duration']:>3d}  conf={seg['conf']:.3f}")

    print(f"\n完成! OK={ok_count}, WR={wr_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="/home/zhaowei/shabi/data/features_v4")
    parser.add_argument("--data-dir", default="/home/zhaowei/shabi/data")
    parser.add_argument("--output-dir", default="/home/zhaowei/shabi/data/features_v4")
    parser.add_argument("--ok-limit", type=int, default=None)
    args = parser.parse_args()
    compute_segmentation_all(args.features_dir, args.data_dir,
                             args.output_dir, args.ok_limit)


if __name__ == "__main__":
    main()
