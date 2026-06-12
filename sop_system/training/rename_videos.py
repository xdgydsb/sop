"""
统一重命名视频及关联文件
ok:  Video_2026...avi → ok_1.avi ... ok_182.avi
wr:  Video_2026...avi → wr_1.avi ... wr_70.avi

同时重命名:
- *_first_frame.jpg
- features_v4/*_X.npy, *_y.npy, *_y_seg.npy (如果存在)

用法:
  # 预览 (不实际执行)
  python rename_videos.py --dry-run

  # 本地执行
  python rename_videos.py

  # 服务器执行
  python rename_videos.py --data-dir /home/zhaowei/shabi/data
"""
import os
from pathlib import Path
import argparse


def build_rename_map(data_dir: str):
    """扫描目录，构建 旧名→新名 映射"""
    data_dir = Path(data_dir)
    rename_map = {}  # old_stem → (old_dir, new_stem)

    for vtype in ["ok", "wr"]:
        vdir = data_dir / vtype
        if not vdir.exists():
            print(f"  SKIP: {vdir} 不存在")
            continue

        videos = sorted(vdir.glob("Video_*.avi"))
        print(f"  {vtype}/: {len(videos)} 个视频")

        for i, v in enumerate(videos, 1):
            old_stem = v.stem  # e.g., "Video_20260510163635161"
            new_stem = f"{vtype}_{i}"
            rename_map[old_stem] = (vdir, new_stem)

    return rename_map


def do_rename(data_dir: str, dry_run: bool = True):
    data_dir = Path(data_dir)
    features_dir = data_dir / "features_v4"

    print("=" * 60)
    print("视频重命名: Video_... → ok_N / wr_N")
    if dry_run:
        print("*** DRY RUN (预览模式) ***")
    print("=" * 60)

    rename_map = build_rename_map(data_dir)
    print(f"\n共 {len(rename_map)} 个视频需要重命名")

    renamed = 0
    errors = []

    for old_stem, (vdir, new_stem) in sorted(rename_map.items()):
        vtype = vdir.name  # "ok" or "wr"

        # 1. 视频 .avi
        old_avi = vdir / f"{old_stem}.avi"
        new_avi = vdir / f"{new_stem}.avi"
        if old_avi.exists():
            if dry_run:
                print(f"  [{vtype}] {old_avi.name} → {new_avi.name}")
            else:
                os.rename(str(old_avi), str(new_avi))
                renamed += 1
        else:
            if not dry_run:
                errors.append(f"MISSING: {old_avi}")

        # 2. first_frame.jpg
        old_jpg = vdir / f"{old_stem}_first_frame.jpg"
        new_jpg = vdir / f"{new_stem}_first_frame.jpg"
        if old_jpg.exists():
            if dry_run:
                print(f"  [{vtype}] {old_jpg.name} → {new_jpg.name}")
            else:
                os.rename(str(old_jpg), str(new_jpg))
        # JPG not critical if missing

        # 3. features_v4/*_X.npy
        old_X = features_dir / f"{old_stem}_X.npy"
        new_X = features_dir / f"{new_stem}_X.npy"
        if old_X.exists():
            if dry_run:
                print(f"  [feat] {old_X.name} → {new_X.name}")
            else:
                os.rename(str(old_X), str(new_X))

        # 4. features_v4/*_y.npy
        old_y = features_dir / f"{old_stem}_y.npy"
        new_y = features_dir / f"{new_stem}_y.npy"
        if old_y.exists():
            if dry_run:
                print(f"  [feat] {old_y.name} → {new_y.name}")
            else:
                os.rename(str(old_y), str(new_y))

        # 5. features_v4/*_y_seg.npy
        old_seg = features_dir / f"{old_stem}_y_seg.npy"
        new_seg = features_dir / f"{new_stem}_y_seg.npy"
        if old_seg.exists():
            if dry_run:
                print(f"  [feat] {old_seg.name} → {new_seg.name}")
            else:
                os.rename(str(old_seg), str(new_seg))

    print(f"\n{'预览' if dry_run else '完成'}! "
          f"{'将重命名' if dry_run else '已重命名'} {renamed} 个视频 + 关联文件")

    if not dry_run:
        # 验证
        for vtype in ["ok", "wr"]:
            vdir = data_dir / vtype
            avi_count = len(list(vdir.glob("*.avi")))
            expected = {"ok": 182, "wr": 70}
            status = "OK" if avi_count == expected.get(vtype, 0) else "FAIL"
            print(f"  {vtype}/: {avi_count} .avi {status}")

        feat_count = len(list(features_dir.glob("*_X.npy")))
        print(f"  features_v4/: {feat_count} _X.npy (期望257)")

    if errors:
        print(f"\n错误: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e}")


def main():
    parser = argparse.ArgumentParser(description="统一重命名视频及关联文件")
    parser.add_argument("--data-dir", default="D:/laji/data")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不执行")
    args = parser.parse_args()

    do_rename(args.data_dir, args.dry_run)


if __name__ == "__main__":
    main()
