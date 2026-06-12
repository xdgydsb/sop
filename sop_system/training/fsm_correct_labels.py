"""
Post-process LSTM auto-labels with FSM constraints.
Forces each video to have S1→S2→S3→S4→S5 in order.
Uses LSTM prediction scores to find best boundaries.
"""
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import argparse
from collections import Counter


def fsm_correct_labels(model, X: np.ndarray, seq_len=48, stride=2, device="cuda",
                       smooth_window=15) -> np.ndarray:
    """
    用LSTM预测分步存在概率，然后FSM强制S1→S2→S3→S4→S5顺序分割。
    """
    N = len(X)
    n_classes = 7

    # 收集每帧的softmax概率 (从所有窗口平均)
    frame_probs = np.zeros((N, n_classes), dtype=np.float64)
    frame_counts = np.zeros(N, dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for start in range(0, N - seq_len, stride):
            end = start + seq_len
            x = torch.FloatTensor(X[start:end]).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            mid = start + seq_len // 2
            # 给窗口中心区域更高权重
            half_w = seq_len // 4
            for fi in range(start + half_w, end - half_w):
                if 0 <= fi < N:
                    frame_probs[fi] += probs
                    frame_counts[fi] += 1

    # 对未覆盖帧使用最近邻居
    for fi in range(N):
        if frame_counts[fi] == 0:
            for d in range(1, N):
                if fi + d < N and frame_counts[fi + d] > 0:
                    frame_probs[fi] = frame_probs[fi + d]
                    frame_counts[fi] = 1
                    break
                if fi - d >= 0 and frame_counts[fi - d] > 0:
                    frame_probs[fi] = frame_probs[fi - d]
                    frame_counts[fi] = 1
                    break

    # 归一化
    for fi in range(N):
        if frame_counts[fi] > 0:
            frame_probs[fi] /= frame_counts[fi]

    # 平滑
    from scipy.ndimage import gaussian_filter1d
    smoothed = np.zeros_like(frame_probs)
    for c in range(n_classes):
        smoothed[:, c] = gaussian_filter1d(frame_probs[:, c], sigma=3.0)

    # FSM强制分段: S1→S2→S3→S4→S5, 用DP找最优边界
    min_seg = 10  # 最短10帧
    max_seg = N // 3  # 最长1/3视频

    # 为每个步骤计算"似然分数" (平滑后的概率 - 其他类的最大概率)
    step_scores = {}
    for s in range(1, 6):
        # 步骤s的得分 = P(s) - max(P(other steps))
        other_mask = np.ones(n_classes, dtype=bool)
        other_mask[s] = False
        other_mask[0] = False
        other_mask[6] = False
        step_scores[s] = smoothed[:, s] - np.max(smoothed[:, other_mask], axis=1)

    # DP: dp[s][end] = max score for step s ending at `end`
    INF_NEG = -1e10
    dp = np.full((6, N), INF_NEG)
    bt = np.zeros((6, N), dtype=np.int32)

    dp[0, :] = 0  # before any step

    for s in range(1, 6):
        score_arr = step_scores[s]
        for end in range(s * min_seg, min(N, s * max_seg)):
            best_score = INF_NEG
            best_start = end - min_seg
            start_min = max((s-1) * min_seg, end - max_seg)
            start_max = end - min_seg + 1
            for start in range(start_min, start_max, 4):  # stride=4
                prev = dp[s-1, start-1] if start > 0 else (0 if s == 1 else INF_NEG)
                if prev <= INF_NEG + 1:
                    continue
                seg_score = np.mean(score_arr[start:end+1])
                score = prev + seg_score
                if score > best_score:
                    best_score = score
                    best_start = start
            dp[s, end] = best_score
            bt[s, end] = best_start

    # 找最佳终点
    best_end = N - 1
    best_total = INF_NEG
    for end in range(5 * min_seg, N):
        if dp[5, end] > best_total:
            best_total = dp[5, end]
            best_end = end

    if best_total <= INF_NEG + 1:
        # DP失败，回退到简单分割
        labels = np.zeros(N, dtype=np.int64)
        seg_size = N // 6
        for s in range(1, 6):
            labels[(s) * seg_size:(s+1) * seg_size] = s
        labels[5*seg_size:] = 5
        return labels

    # 回溯
    ends = [0] * 5
    starts = [0] * 5
    cur_end = best_end
    for s in range(5, 0, -1):
        ends[s-1] = cur_end
        starts[s-1] = int(bt[s, cur_end])
        cur_end = starts[s-1] - 1

    # 生成标签
    labels = np.zeros(N, dtype=np.int64)
    for s in range(1, 6):
        start, end = max(0, starts[s-1]), min(N-1, ends[s-1])
        if end >= start:
            labels[start:end+1] = s

    # 步骤间填充: 过渡帧用前后步骤的众数
    for s in range(1, 6):
        if s < 5:
            gap_start = ends[s-1] + 1
            gap_end = starts[s] - 1
        else:
            gap_start = ends[s-1] + 1
            gap_end = N - 1
        if gap_start <= gap_end:
            for fi in range(gap_start, gap_end + 1):
                labels[fi] = s if fi < N // 2 else (s + 1 if s < 5 else s)

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="/home/zhaowei/shabi/data/features_v4")
    parser.add_argument("--model-path", default="/home/zhaowei/shabi/sop_system/runs/best_lstm.pt")
    parser.add_argument("--start", type=int, default=71)
    parser.add_argument("--end", type=int, default=182)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from engine.temporal_lstm import SOPActionLSTM

    features_dir = Path(args.features_dir)

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=True)
    actual_hidden = ckpt["model_state_dict"]["lstm.weight_ih_l0"].shape[0] // 4
    print(f"Model: hidden_size={actual_hidden}, val_acc={ckpt['val_acc']:.3f}")

    model = SOPActionLSTM(
        input_size=ckpt["input_size"],
        hidden_size=actual_hidden,
        num_layers=ckpt["num_layers"],
        num_classes=ckpt["num_classes"],
        dropout=ckpt.get("dropout", 0.4),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    good = 0
    for idx in range(args.start, args.end + 1):
        x_path = features_dir / f"ok_{idx}_X.npy"
        y_path = features_dir / f"ok_{idx}_y_seg.npy"
        if not x_path.exists():
            continue

        X = np.load(x_path).astype(np.float32)
        if len(X) < 48:
            continue

        labels = fsm_correct_labels(model, X, device=args.device)
        np.save(y_path, labels)

        counts = {s: int(np.sum(labels == s)) for s in range(0, 7)}
        found = [s for s in range(1, 6) if counts[s] > 0]
        all5 = set(found) == {1, 2, 3, 4, 5}
        if all5:
            good += 1

        if idx <= 75 or idx % 20 == 0 or all5:
            status = "OK" if all5 else "partial"
            print(f"  ok_{idx}: {len(X)}f, steps={found} {status} | "
                  f"S0={counts[0]} S1={counts[1]} S2={counts[2]} "
                  f"S3={counts[3]} S4={counts[4]} S5={counts[5]}")

    print(f"\nFSM-corrected: {good}/{args.end - args.start + 1} videos have all 5 steps")
    print("Now ready for LSTM retraining on all 182 videos.")


if __name__ == "__main__":
    main()
