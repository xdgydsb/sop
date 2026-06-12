"""
离线测试 — 使用 features_v7 数据评测 BiGRU + EventDetector + FSM

测试方法 (对齐 realtime_pipeline 逻辑):
  1. 滑动窗口推理 (128-frame 窗口, stride=8, 匹配训练配置)
  2. EMA 平滑模型输出
  3. EventDetector: stability counting + one-shot + confidence threshold
  4. FSM: 序列验证
  5. 指标: 逐帧准确率, 段边界召回率, OK/WR FSM 通过率

Usage:
  python offline_test.py                  # 全部 250 文件
  python offline_test.py --quick          # 快速 20 文件
  python offline_test.py --single ok_100  # 单文件详情
"""
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from engine.sop_fsm import SOPStateMachine, SOPStep, STEP_NAMES
from engine.temporal_lstm import SOPActionGRU

# ── 常量 ──
FEATURES_DIR = Path(__file__).parent / "data" / "features_v7"
LABELS_DIR = Path(__file__).parent / "data" / "labels_correct"
MODEL_PATH = Path(__file__).parent / "models" / "best_sequence_v5.pt"

# 推理参数 (对齐训练配置)
WINDOW_SIZE = 128   # 训练时的 SEQ_LEN
WINDOW_STRIDE = 8   # 滑动步长

STEP_TO_NAME = {
    0: "背景", 1: "S1-打开纸盒", 2: "S2-放入耳机盒",
    3: "S3-放入插头", 4: "S4-放入绿袋", 5: "S5-关闭纸盒", 6: "完成",
}

OBJ_TO_STEP = {"earphone": 2, "charger": 3, "green_bag": 4}


def load_model() -> SOPActionGRU:
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = SOPActionGRU(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        num_classes=ckpt["num_classes"],
        dropout=0.4,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Model] BiGRU input={ckpt['input_size']}, hidden={ckpt['hidden_size']}, "
          f"layers={ckpt['num_layers']}, val_acc={ckpt['val_acc']:.1%}")
    return model


def load_feature_files(category: str = "all") -> List[Tuple[str, str]]:
    pairs = []
    for f in sorted(FEATURES_DIR.glob("*_X.npy")):
        name = f.stem.replace("_X", "")
        if name.startswith("ok_"):
            cat = "ok"
        elif name.startswith("wr_"):
            cat = "wr"
        else:
            cat = "unknown"
        y_file = FEATURES_DIR / f"{name}_y_seg.npy"
        if not y_file.exists():
            y_file = FEATURES_DIR / f"{name}_y.npy"
        if not y_file.exists():
            continue
        if category == "all" or category == cat:
            pairs.append((name, cat))
    return pairs


def sliding_window_inference(model: SOPActionGRU, X: np.ndarray,
                             window_size: int = WINDOW_SIZE,
                             stride: int = WINDOW_STRIDE) -> np.ndarray:
    """
    滑动窗口推理 — 每个窗口独立前向传播，重叠帧取平均概率。
    返回: (N_frames, 7) 概率数组
    """
    N = len(X)
    probs_accum = np.zeros((N, 7), dtype=np.float64)
    counts = np.zeros(N, dtype=np.int32)

    for start in range(0, max(1, N - window_size + stride), stride):
        end = min(start + window_size, N)
        seg = X[start:end]

        if len(seg) < window_size // 4:
            continue

        # 如果不够 window_size，pad with edge values
        if len(seg) < window_size:
            pad_len = window_size - len(seg)
            seg = np.pad(seg, ((0, pad_len), (0, 0)), mode='edge')

        x = torch.FloatTensor(seg).unsqueeze(0)  # (1, W, 130)
        with torch.no_grad():
            logits = model(x)  # (1, W, 7)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()  # (W, 7)

        # 只取实际帧的预测 (去掉 padding)
        actual_len = min(window_size, N - start)
        probs_accum[start:start + actual_len] += probs[:actual_len]
        counts[start:start + actual_len] += 1

    # 处理未覆盖的帧 (开头和结尾)
    for i in range(N):
        if counts[i] == 0:
            # 用最近的有效帧填充
            for d in range(1, N):
                if i + d < N and counts[i + d] > 0:
                    probs_accum[i] = probs_accum[i + d]
                    counts[i] = 1
                    break
                if i - d >= 0 and counts[i - d] > 0:
                    probs_accum[i] = probs_accum[i - d]
                    counts[i] = 1
                    break

    # 归一化
    for i in range(N):
        if counts[i] > 0:
            probs_accum[i] /= counts[i]

    return probs_accum.astype(np.float32)


def ema_smooth_probs(probs: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """EMA 平滑逐帧概率"""
    smoothed = np.zeros_like(probs)
    smoothed[0] = probs[0]
    for i in range(1, len(probs)):
        smoothed[i] = alpha * probs[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


class EventDetectorSim:
    """
    离线 EventDetector 模拟 (对齐 realtime_pipeline._detect_event)

    规则:
      Rule 0: 错误物体检测 (from labels — 跳过此步在离线测试)
      Rule 1: box_closed + 所有物体入盒 → S5
      Rule 2: box_open + idle → S1
      Rule 3: box_open + 模型预测稳定 → S2-S4
      Rule 4: 默认 → 保持

    事件 one-shot: 每个步骤只触发一次
    稳定性确认: 连续 stable_frames 帧同一预测
    """

    def __init__(self, conf_threshold: float = 0.45,
                 stable_frames: int = 8, cooldown: int = 20):
        self.conf_threshold = conf_threshold
        self.stable_frames = stable_frames
        self.cooldown = cooldown

        self._emitted: set = set()
        self._stable_cnt: Dict[int, int] = defaultdict(int)
        self._last_event_frame: int = -cooldown
        self.events: List[Dict] = []  # [{frame, step, conf, source}]

    def reset(self):
        self._emitted.clear()
        self._stable_cnt.clear()
        self._last_event_frame = -self.cooldown
        self.events.clear()

    def update(self, frame_idx: int, model_probs: np.ndarray,
               phys_step: int) -> Optional[int]:
        """
        返回触发的事件步骤 (1-6)，或 None。

        Args:
            frame_idx: 当前帧索引
            model_probs: (7,) EMA 平滑后的模型概率
            phys_step: 从标签推导的当前物理步骤
        """
        model_step = int(np.argmax(model_probs))
        model_conf = float(model_probs[model_step])

        # 只处理有效步骤 (S1-S6)
        if model_step < 1 or model_step > 6:
            return None
        if model_conf < self.conf_threshold:
            return None

        # 已触发过
        if model_step in self._emitted:
            return None

        # 稳定性计数 (只有当前预测持续时才增加)
        self._stable_cnt[model_step] += 1
        for k in list(self._stable_cnt.keys()):
            if k != model_step:
                self._stable_cnt[k] = max(0, self._stable_cnt[k] - 1)

        # 触发检查
        if (self._stable_cnt[model_step] >= self.stable_frames
                and frame_idx - self._last_event_frame >= self.cooldown):

            # 物理校验: 模型超前物理状态太多 → 不触发
            if model_step > phys_step + 2:
                return None

            self._emitted.add(model_step)
            self._last_event_frame = frame_idx

            event = {
                "frame": frame_idx,
                "step": model_step,
                "conf": model_conf,
                "source": "model+stable",
            }
            self.events.append(event)
            return model_step

        return None


def evaluate_file(model: SOPActionGRU, name: str, cat: str,
                  event_detector: EventDetectorSim,
                  verbose: bool = False) -> Dict:
    """评测单个文件 — 完整 pipeline 模拟"""
    X = np.load(FEATURES_DIR / f"{name}_X.npy")
    y_file = FEATURES_DIR / f"{name}_y_seg.npy"
    if not y_file.exists():
        y_file = FEATURES_DIR / f"{name}_y.npy"
    y_true = np.load(y_file)
    n_frames = len(X)

    # ── 1. 滑动窗口推理 ──
    raw_probs = sliding_window_inference(model, X)

    # ── 2. EMA 平滑 ──
    smooth_probs = ema_smooth_probs(raw_probs, alpha=0.35)
    y_pred = np.argmax(smooth_probs, axis=1)

    # ── 3. 逐帧准确率 ──
    correct = (y_pred == y_true).sum()
    acc = correct / n_frames

    per_step_acc = {}
    for s in range(7):
        mask = y_true == s
        if mask.sum() > 0:
            per_step_acc[s] = (y_pred[mask] == s).sum() / mask.sum()

    # ── 4. 混淆矩阵 ──
    confusion = np.zeros((7, 7), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    # ── 5. EventDetector + FSM 模拟 ──
    fsm = SOPStateMachine(timeout=90.0)
    event_detector.reset()

    errors = []
    fsm_steps = []  # FSM 记录的状态转移

    for i in range(n_frames):
        phys_step = int(y_true[i])

        # EventDetector
        triggered = event_detector.update(i, smooth_probs[i], phys_step)

        # 如果有事件触发, 驱动 FSM
        if triggered is not None:
            fsm_result = fsm.validate(
                triggered,
                float(smooth_probs[i][triggered]),
                physical_state_ok=True,
                timestamp=float(i) / 30.0,
            )
            if fsm_result.has_error:
                errors.append({
                    "frame": i,
                    "triggered_step": triggered,
                    "error_type": fsm_result.error_type,
                    "message": fsm_result.message,
                })
            if fsm_result.step_id != fsm.current_step.value:
                # 记录状态变化
                pass

    # 收集 FSM 路径
    fsm_path = [h["step"] for h in fsm.step_history]

    # ── 6. 段边界检测 ──
    true_transitions = []
    prev = -1
    for i, s in enumerate(y_true):
        if s != prev and s > 0:
            true_transitions.append((i, int(s)))
        prev = int(s)

    pred_transitions = []
    prev = -1
    for i, s in enumerate(y_pred):
        if s != prev and s > 0:
            pred_transitions.append((i, int(s)))
        prev = int(s)

    event_transitions = [(e["frame"], e["step"]) for e in event_detector.events]

    # 边界容差匹配 (±30帧 ≈ 1秒)
    tol = 30
    boundary_hits = 0
    for t_frame, t_step in true_transitions:
        for p_frame, p_step in event_transitions:
            if t_step == p_step and abs(t_frame - p_frame) <= tol:
                boundary_hits += 1
                break

    # ── 7. 汇总 ──
    true_steps = sorted(set(int(s) for s in y_true if s > 0))
    pred_steps_set = sorted(set(int(s) for s in y_pred if s > 0))
    event_steps = sorted(set(e["step"] for e in event_detector.events))

    if verbose:
        print(f"\n{'='*70}")
        print(f"File: {name}  |  {cat}  |  {n_frames} frames")
        print(f"{'='*70}")
        print(f"  Per-frame acc: {acc:.1%} ({correct}/{n_frames})")
        print(f"  Per-step acc: " + " | ".join(
            f"S{s}: {per_step_acc.get(s, 0):.1%}" for s in range(7) if s in per_step_acc))
        print(f"  True steps in data: {true_steps}")
        print(f"  Model predicted steps: {pred_steps_set}")
        print(f"  EventDetector events: {event_transitions}")
        print(f"  True transitions: {true_transitions}")
        print(f"  Boundary recall (±{tol}f): {boundary_hits}/{len(true_transitions)} "
              f"({boundary_hits/max(len(true_transitions),1):.1%})")
        print(f"  FSM path: {fsm_path}")
        print(f"  FSM final step: {fsm.current_step.value} ({STEP_NAMES.get(SOPStep(fsm.current_step.value) if fsm.current_step.value in [e.value for e in SOPStep] else SOPStep.ERROR, '?')})")
        if errors:
            print(f"  FSM errors ({len(errors)}):")
            for e in errors[:5]:
                print(f"    [{e['frame']}] {e['error_type']}: {e['message']}")
        else:
            print(f"  FSM errors: 0")

    return {
        "name": name, "category": cat, "n_frames": n_frames,
        "accuracy": acc, "per_step_acc": per_step_acc, "confusion": confusion,
        "true_steps": true_steps, "pred_steps": pred_steps_set,
        "event_steps": event_steps, "event_transitions": event_transitions,
        "true_transitions": true_transitions,
        "boundary_hits": boundary_hits, "n_boundaries": len(true_transitions),
        "fsm_errors": errors, "fsm_error_count": len(errors),
        "fsm_final_step": fsm.current_step.value, "fsm_path": fsm_path,
    }


def main():
    parser = argparse.ArgumentParser(description="SOP离线测试")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--single", type=str)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.45,
                        help="EventDetector 置信度阈值")
    parser.add_argument("--stable", type=int, default=8,
                        help="EventDetector 稳定帧数")
    args = parser.parse_args()

    print("=" * 70)
    print("SOP 离线测试 — SlidingWindow + EMA + EventDetector + FSM")
    print(f"  Window: {WINDOW_SIZE}f  Stride: {WINDOW_STRIDE}f")
    print(f"  EventDetector: conf≥{args.conf}, stable={args.stable}f")
    print("=" * 70)

    model = load_model()

    if args.single:
        file_list = [(args.single, "ok" if args.single.startswith("ok") else "wr")]
    else:
        file_list = load_feature_files("all")
        if args.quick:
            file_list = file_list[:20]
        elif args.max_files > 0:
            file_list = file_list[:args.max_files]

    n_ok = sum(1 for _, c in file_list if c == "ok")
    n_wr = sum(1 for _, c in file_list if c == "wr")
    print(f"[Data] {len(file_list)} files (OK={n_ok}, WR={n_wr})")

    event_detector = EventDetectorSim(
        conf_threshold=args.conf,
        stable_frames=args.stable,
        cooldown=20,
    )

    all_results = []
    ok_results = []
    wr_results = []
    total_confusion = np.zeros((7, 7), dtype=int)
    total_correct = 0
    total_frames = 0

    t_start = time.time()
    for name, cat in file_list:
        result = evaluate_file(
            model, name, cat, event_detector,
            verbose=bool(args.single)
        )
        all_results.append(result)
        total_correct += int(result["accuracy"] * result["n_frames"])
        total_frames += result["n_frames"]
        total_confusion += result["confusion"]

        if cat == "ok":
            ok_results.append(result)
        else:
            wr_results.append(result)

    elapsed = time.time() - t_start
    print(f"\n[Timing] {len(file_list)} files in {elapsed:.1f}s "
          f"({elapsed/max(len(file_list),1):.2f}s/file)")

    # ──────── 汇总报告 ────────
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    # 1. 总体准确率
    overall_acc = total_correct / max(total_frames, 1)
    print(f"\n  Overall per-frame accuracy: {overall_acc:.1%} "
          f"({total_correct:,}/{total_frames:,})")

    # 2. 混淆矩阵
    print("\n  Confusion Matrix (row=true, col=pred):")
    print("        " + " ".join(f"  P{s}" for s in range(7)))
    for t in range(7):
        row = f"    T{t}: " + " ".join(f"{total_confusion[t,p]:4d}" for p in range(7))
        print(row)

    print("\n  Per-class recall/precision:")
    print(f"  {'Class':<20} {'Recall':>8} {'Precision':>10} {'Support':>8}")
    for s in range(7):
        tp = total_confusion[s, s]
        sup = total_confusion[s, :].sum()
        pred_tot = total_confusion[:, s].sum()
        rec = tp / max(sup, 1)
        prec = tp / max(pred_tot, 1)
        print(f"  {STEP_TO_NAME.get(s, f'C{s}'):<20} {rec:>7.1%} {prec:>9.1%} {sup:>8}")

    # 3. OK/WR 准确率
    if ok_results:
        ok_acc = sum(r["accuracy"] * r["n_frames"] for r in ok_results) / max(
            sum(r["n_frames"] for r in ok_results), 1)
        print(f"\n  OK accuracy: {ok_acc:.1%} ({len(ok_results)} files)")
    if wr_results:
        wr_acc = sum(r["accuracy"] * r["n_frames"] for r in wr_results) / max(
            sum(r["n_frames"] for r in wr_results), 1)
        print(f"  WR accuracy: {wr_acc:.1%} ({len(wr_results)} files)")

    # 4. 边界召回
    total_hits = sum(r["boundary_hits"] for r in all_results)
    total_boundaries = sum(r["n_boundaries"] for r in all_results)
    print(f"\n  Boundary recall (±30f): {total_hits}/{total_boundaries} "
          f"({total_hits/max(total_boundaries,1):.1%})")

    # 5. FSM 路径分析
    print(f"\n  ── FSM Path Analysis ──")
    # OK: 应该形成完整序列 S1→S2→S3→S4→S5
    ok_fsm_complete = sum(
        1 for r in ok_results
        if set(r["fsm_path"]) >= {1, 2, 3, 4, 5}
        or (r["fsm_error_count"] == 0 and r["fsm_final_step"] >= 5)
    )
    print(f"  OK complete path (S1-S5): {ok_fsm_complete}/{len(ok_results)}")

    ok_fsm_errors = sum(1 for r in ok_results if r["fsm_error_count"] > 0)
    print(f"  OK with FSM errors: {ok_fsm_errors}/{len(ok_results)} "
          f"({ok_fsm_errors/max(len(ok_results),1):.1%})")

    # WR: 应该有错误被检测
    if wr_results:
        wr_with_errors = sum(1 for r in wr_results if r["fsm_error_count"] > 0)
        print(f"  WR with FSM errors detected: {wr_with_errors}/{len(wr_results)} "
              f"({wr_with_errors/max(len(wr_results),1):.1%})")

        # 分析WR的标签模式
        print(f"\n  ── WR Label Patterns (missing steps) ──")
        pattern_cnt = Counter()
        for r in wr_results:
            missing = tuple(s for s in [1, 2, 3, 4, 5] if s not in r["true_steps"])
            pattern_cnt[missing] += 1
        for pat, cnt in pattern_cnt.most_common():
            missing_names = [f"S{s}" for s in pat]
            print(f"    missing {missing_names}: {cnt} files")

    # 6. EventDetector 事件分析
    print(f"\n  ── EventDetector Events ──")
    ok_events_per_file = []
    for r in ok_results:
        if r["event_transitions"]:
            ok_events_per_file.append(len(r["event_transitions"]))
    if ok_events_per_file:
        print(f"  OK events/file: mean={np.mean(ok_events_per_file):.1f}, "
              f"median={np.median(ok_events_per_file):.0f}, "
              f"min={np.min(ok_events_per_file)}, max={np.max(ok_events_per_file)}")

    # OK 文件的事件步骤分布
    ok_event_step_cnt = Counter()
    for r in ok_results:
        for s in r["event_steps"]:
            ok_event_step_cnt[s] += 1
    print(f"  OK event steps: {dict(sorted(ok_event_step_cnt.items()))}")

    # 7. 准确率分布
    accs = [r["accuracy"] for r in all_results]
    print(f"\n  ── Accuracy Distribution ──")
    print(f"  Mean={np.mean(accs):.1%}  Median={np.median(accs):.1%}  "
          f"Std={np.std(accs):.2%}  Min={np.min(accs):.1%}  Max={np.max(accs):.1%}")
    worst = sorted(all_results, key=lambda r: r["accuracy"])[:5]
    print(f"  Worst 5:")
    for r in worst:
        print(f"    {r['name']} ({r['category']}): {r['accuracy']:.1%} "
              f"({r['n_frames']}f, steps={r['true_steps']})")
    best = sorted(all_results, key=lambda r: -r["accuracy"])[:5]
    print(f"  Best 5:")
    for r in best:
        print(f"    {r['name']} ({r['category']}): {r['accuracy']:.1%} "
              f"({r['n_frames']}f, steps={r['true_steps']})")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
