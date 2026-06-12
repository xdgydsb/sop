"""
从视频中采样200帧用于手工标注
策略: 均匀覆盖所有视频+所有动作阶段，重点采样charger/green_bag出现时段
"""
import cv2
import numpy as np
from pathlib import Path
import sys

VIDEO_DIR = Path("/home/zhaowei/shabi/data")
OUTPUT_DIR = Path("/home/zhaowei/shabi/data/manual_label_frames")
N_FRAMES = 200


def sample_frames():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 视频来源
    sources = []

    # yuanzi单动作视频 (5个，每个采样10帧)
    for i in range(1, 6):
        vp = VIDEO_DIR / "yuanzi" / f"{i}.avi"
        if vp.exists():
            sources.append((vp, "yuanzi", 12))

    # OK视频 (采样~130帧，分散到不同视频)
    ok_videos = sorted((VIDEO_DIR / "ok").glob("*.avi"))
    n_ok = min(40, len(ok_videos))
    for vp in ok_videos[:n_ok]:
        sources.append((vp, "ok", max(1, 130 // n_ok)))

    # WR视频 (采样~40帧)
    wr_videos = sorted((VIDEO_DIR / "wr").glob("*.avi"))
    n_wr = min(15, len(wr_videos))
    for vp in wr_videos[:n_wr]:
        sources.append((vp, "wr", max(1, 40 // n_wr)))

    print(f"Sources: {len(sources)} videos, target {N_FRAMES} frames")

    frame_idx = 0
    for vp, tag, n_sample in sources:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"  SKIP: {vp}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 10:
            cap.release()
            continue

        # 均匀采样: 避免开头结尾空白
        start = int(total_frames * 0.05)
        end = int(total_frames * 0.95)
        if end - start < n_sample:
            indices = np.linspace(start, end, min(n_sample, end - start), dtype=int)
        else:
            indices = np.linspace(start, end, n_sample, dtype=int)

        video_name = vp.stem
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue

            out_name = f"{frame_idx:04d}_{tag}_{video_name}_f{fi}.jpg"
            cv2.imwrite(str(OUTPUT_DIR / out_name), frame)
            frame_idx += 1

            if frame_idx >= N_FRAMES:
                cap.release()
                print(f"\nDone! Exported {frame_idx} frames to {OUTPUT_DIR}")
                return

        cap.release()

    print(f"\nDone! Exported {frame_idx} frames to {OUTPUT_DIR}")


if __name__ == "__main__":
    sample_frames()
