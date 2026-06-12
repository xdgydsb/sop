"""
SOP Video Annotation Tool
Mark start/end frame for each of the 5 actions per video.

Usage: python annotate_video.py ok 1
       python annotate_video.py wr 1

Controls:
  1-5  Press once = mark START of action S1-S5
       Press again = mark END of that action
       Press 3rd time = clear and re-mark

  A/D    -1/+1 frame
  W/S    -30/+30 frames
  Space  Play/Pause
  +/-    Speed up/down
  Z      Undo last mark
  R      Reset all
  P      Save to _y_seg.npy
  Q      Quit
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# BGR colors
C = {
    0: (100, 100, 100),
    1: (60, 60, 255),      # red - S1 Open box
    2: (60, 255, 60),      # green - S2 Earphone
    3: (60, 180, 255),     # orange - S3 Charger
    4: (220, 80, 255),     # purple - S4 Green bag
    5: (255, 180, 60),     # blue - S5 Close box
}

STEP_LABEL = {0:"BG", 1:"S1-OpenBox", 2:"S2-Earphone", 3:"S3-Charger", 4:"S4-GreenBag", 5:"S5-CloseBox"}


class Annotator:
    def __init__(self, video_path, save_path=None):
        self.cap = cv2.VideoCapture(str(video_path))
        self.N = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
        self.save_path = Path(save_path) if save_path else None

        # For each step 1..5: [start_frame, end_frame] or None
        self.seg = {s: None for s in range(1, 6)}
        # Pending start (second press = confirm end)
        self.pending = {}

        if self.save_path and self.save_path.exists():
            self._load()

        self.pos = 0
        self.img = None
        self.playing = False
        self.ms = 25
        self._seek(0)

    def _seek(self, idx):
        self.pos = max(0, min(self.N - 1, idx))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.pos)
        ok, f = self.cap.read()
        if ok:
            self.img = f

    # ---- labels ----
    def labels(self):
        y = np.zeros(self.N, dtype=np.int64)
        for s in range(1, 6):
            sg = self.seg[s]
            if sg is not None:
                a, b = sg
                y[max(0, a):min(self.N, b+1)] = s
        return y

    def _save(self):
        if not self.save_path:
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.save_path), self.labels())
        print(f"\n>>> Saved: {self.save_path.name}")
        for s in range(1, 6):
            sg = self.seg[s]
            if sg is not None:
                a, b = sg
                dur = (b - a + 1) / self.FPS
                print(f"  {STEP_LABEL[s]:15s}  [{a:5d} -> {b:5d}]  {b-a+1:4d}f  {dur:.1f}s")
        # Print step sequence (for WR checking)
        ordered = sorted([(sg[0], s) for s, sg in self.seg.items() if sg is not None])
        seq = " -> ".join(f"S{s}" for _, s in ordered)
        print(f"  Sequence: {seq}")

    def _load(self):
        y = np.load(str(self.save_path)).astype(np.int64)
        for s in range(1, 6):
            idxs = np.where(y == s)[0]
            if len(idxs) > 0:
                self.seg[s] = [int(idxs[0]), int(idxs[-1])]
        print(f"  Loaded: {sum(1 for s in range(1,6) if self.seg[s] is not None)}/5 actions")

    # ---- step key handler ----
    def _step_key(self, s):
        if s in self.pending:
            # 2nd press: set END
            a = self.pending.pop(s)
            b = self.pos
            if b < a:
                a, b = b, a
            self.seg[s] = [a, b]
            print(f"  {STEP_LABEL[s]:15s} START={a:5d}  END={b:5d}  DUR={b-a+1}f")

        elif self.seg[s] is not None:
            # 3rd press: redo
            old = self.seg[s]
            self.seg[s] = None
            self.pending[s] = self.pos
            print(f"  {STEP_LABEL[s]:15s} RE-MARK  old={old}  new START={self.pos}")

        else:
            # 1st press: set START
            self.pending[s] = self.pos
            print(f"  {STEP_LABEL[s]:15s} START={self.pos}  (navigate then press {s} again)")

        # Auto-save when all 5 done
        done = sum(1 for s in range(1, 6) if self.seg[s] is not None)
        if done == 5 and not self.pending and self.save_path:
            self._save()

    # ---- draw ----
    def _draw(self):
        h, w = self.img.shape[:2]
        TH = 80  # timeline height
        can = np.zeros((h + TH + 35, w, 3), dtype=np.uint8)
        can[TH:TH+h, :w] = self.img

        Y = self.labels()

        # Timeline bars
        for x in range(w):
            fi = int(x / max(1, w-1) * (self.N-1))
            fi = min(fi, self.N-1)
            can[25:TH-5, x] = C.get(int(Y[fi]), (80, 80, 80))

        # Completed segments: bright fill
        for s in range(1, 6):
            sg = self.seg[s]
            if sg is None:
                continue
            sx = int(sg[0] / max(1, self.N-1) * (w-1))
            ex = int(sg[1] / max(1, self.N-1) * (w-1))
            cv2.rectangle(can, (sx, 28), (ex, TH-8), C[s], -1)
            cv2.rectangle(can, (sx, 28), (ex, TH-8), (255, 255, 255), 1)
            cv2.putText(can, f"S{s}", (sx+2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(can, f"S{s}", (ex-20, TH-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # Pending: yellow line from start to current
        for s, pstart in self.pending.items():
            sx = int(pstart / max(1, self.N-1) * (w-1))
            cx = int(self.pos / max(1, self.N-1) * (w-1))
            cv2.line(can, (sx, 28), (sx, TH-8), C[s], 3)
            cv2.line(can, (sx, TH//2), (cx, TH//2), (0, 255, 255), 2)
            cv2.putText(can, f"S{s}...", (sx+2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        # Cursor line
        cx = int(self.pos / max(1, self.N-1) * (w-1))
        cv2.line(can, (cx, 0), (cx, TH+h), (0, 255, 255), 2)

        # Bottom info bar
        info = []
        info.append(f"Frame: {self.pos}/{self.N} ({100*self.pos//max(1,self.N)}%)")
        info.append(f"Label: {STEP_LABEL[int(Y[self.pos])]}")
        for s in range(1, 6):
            sg = self.seg[s]
            if sg is not None:
                info.append(f"S{s}:[{sg[0]},{sg[1]}]")
            elif s in self.pending:
                info.append(f"S{s}:[{self.pending[s]},?]")
        if self.playing:
            info.append("PLAYING")

        y0 = TH + h + 12
        cv2.putText(can, " | ".join(info), (5, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Right panel: step summary
        x0 = w - 280
        cv2.rectangle(can, (x0, TH), (w, TH+135), (35, 35, 35), -1)
        for i, s in enumerate(range(1, 6)):
            y = TH + 8 + i * 25
            sg = self.seg[s]
            if sg is not None:
                txt = f"S{s} [{sg[0]}->{sg[1]}] {sg[1]-sg[0]+1}f"
            elif s in self.pending:
                txt = f"S{s} START={self.pending[s]} ..."
            else:
                txt = f"S{s}  --"
            cv2.putText(can, txt, (x0+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C[s], 1)

        # Help overlay (bottom-right)
        help_lines = ["A/D:+/-1f  W/S:+/-30f", "Space:Play  P:Save  Q:Quit"]
        for i, line in enumerate(help_lines):
            cv2.putText(can, line, (5, y0+15+i*14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)

        return can

    # ---- main loop ----
    def run(self):
        wname = f"Annotator - {self.save_path.stem if self.save_path else '?'}"
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wname, 1280, 800)

        print(f"\n{'='*55}")
        print(f"Video: {self.save_path.stem if self.save_path else '?'}")
        print(f"Frames: {self.N}  Duration: {self.N/self.FPS:.1f}s  FPS: {self.FPS:.0f}")
        print(f"{'='*55}")
        print("Press 1-5 TWICE per action: 1st=START, 2nd=END, 3rd=REDO")
        print("A/D=-/+1f  W/S=-/+30f  Space=Play  P=Save  Q=Quit")

        while True:
            can = self._draw()
            cv2.imshow(wname, can)

            if self.playing:
                raw = cv2.waitKeyEx(self.ms)
                if raw == -1:
                    self._seek(self.pos + (1 if self.pos < self.N-1 else 0))
                    if self.pos >= self.N - 1:
                        self.playing = False
                    continue
            else:
                raw = cv2.waitKeyEx(0)
                if raw == -1:
                    continue

            k = raw & 0xFF

            # Step keys 1-5
            if ord('1') <= k <= ord('5'):
                self._step_key(k - ord('0'))

            # Navigation
            elif k == ord('a'):   self._seek(self.pos - 1)
            elif k == ord('d'):   self._seek(self.pos + 1)
            elif k == ord('w'):   self._seek(self.pos - 30)
            elif k == ord('s'):   self._seek(self.pos + 30)

            elif k == 32:  self.playing = not self.playing

            elif k == ord('+') or k == ord('='):  self.ms = max(5, self.ms - 10)
            elif k == ord('-'):                    self.ms = min(200, self.ms + 10)

            elif k == ord('z'):
                if self.pending:
                    s = list(self.pending.keys())[-1]
                    print(f"  Undo: cancel S{s} pending")
                    del self.pending[s]
                elif any(self.seg[s] is not None for s in range(1, 6)):
                    s = max(s for s in range(1, 6) if self.seg[s] is not None)
                    print(f"  Undo: clear S{s}")
                    self.seg[s] = None

            elif k == ord('r'):
                self.seg = {s: None for s in range(1, 6)}
                self.pending = {}
                print("  Reset all")

            elif k == ord('p') or k == 13:
                self._save()

            elif k == ord('q') or k == 27:
                done = sum(1 for s in range(1, 6) if self.seg[s] is not None)
                if done > 0:
                    ans = input(f"\n  {done}/5 marked. Save before quit? (y/n) ").strip().lower()
                    if ans == 'y':
                        self._save()
                self.cap.release()
                cv2.destroyAllWindows()
                return


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    vtype = sys.argv[1]
    idx_str = sys.argv[2] if len(sys.argv) > 2 else "1"
    base = Path("D:/laji/data")
    fdir = base / "features_v4"
    fdir.mkdir(parents=True, exist_ok=True)

    if "-" in idx_str:
        a, b = idx_str.split("-")
        indices = list(range(int(a), int(b) + 1))
    else:
        indices = [int(idx_str)]

    for idx in indices:
        vname = f"{vtype}_{idx}" if vtype in ("ok", "wr") else str(idx)
        vpath = base / vtype / f"{vname}.avi"
        spath = fdir / f"{vname}_y_seg.npy"

        if not vpath.exists():
            print(f"ERROR: not found -> {vpath}")
            continue

        a = Annotator(str(vpath), str(spath))
        a.run()

        if idx != indices[-1]:
            ans = input(f"\nNext: {vtype}_{indices[indices.index(idx)+1]}? (y/n) ").strip().lower()
            if ans != 'y':
                break


if __name__ == "__main__":
    main()
