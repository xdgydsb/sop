# SOP Detect Stable Checkpoint

Date: 2026-06-11 19:58:31 +08:00

This checkpoint records the first version that the user confirmed as meeting the current SOP detection needs.

## User-Confirmed Status

- Current version is acceptable: "可以了，目前满足我的需求了".
- Keep this as the stable baseline before any later changes.
- Do not make broad rewrites. Future fixes should be small, targeted, and verified against this checkpoint.

## Stable Runtime Topology

- RuoYi frontend: `http://localhost/sop/detect`
- Frontend process: `node ... vue-cli-service.js serve --port 80`
- RuoYi backend: `http://127.0.0.1:8080`
- Backend process: `java -jar ruoyi-admin.jar`
- Local camera bridge: `http://127.0.0.1:18081`
- Bridge process:
  `D:\Anaconda\python.exe ruoyi_hik_ws_bridge.py --server 192.168.31.19 --port 8765 --product-id 100 --sop-id 100 --ruoyi-base-url http://127.0.0.1:8080 --public-host 127.0.0.1 --station-code STATION-01 --camera-code MV-CS050-10UC`
- Remote inference server: `zhaowei@192.168.31.19:22`, stage3 on port `8765`
- Camera: Hikrobot USB industrial camera `MV-CS050-10UC`

## Verified Stable Files

- `D:\gsdcs\sop_system\ruoyi_hik_ws_bridge.py`
  SHA256: `480C67EC2A5667F0DE837FD5DE70B5466BA298917AB42927F6BC242ECAB6ED2F`
- `D:\gsdcs\SOP前后端\ruoyi-ui\src\views\sop\detect\index.vue`
  SHA256: `008EB46D308D683EE40BADC85A3BCFEF624F2EC76551469E520CEAFC6545A8D6`

## Key Fixes Preserved Here

- Real-time camera page uses the Hikrobot live bridge and server stage3 inference, not offline video.
- Main live view uses MJPEG first and falls back to refreshed `latest.jpg` if the stream stalls.
- Start/stop/reset remain button-driven; no auto-start assumptions.
- Reset clears visible step state immediately back to `PENDING`.
- Step clips are generated only from the corresponding accepted step event.
- The bridge no longer creates fake identical S1-S5 clips from one batched server response.
- Generated clip URLs are synced to RuoYi immediately after MP4 writing completes.

## Detection Rules That Must Not Regress

- Do not replace stage3 with stage1.
- Do not rewrite YOLO or the established stage3 detection core unless a specific verified defect requires it.
- S1 must be a real hand-driven box transition from closed to open.
- Object/hand occlusion over a closed box must not count as S1.
- S2/S3/S4 require the item to move from outside into the opened box area and remain there.
- Directly placing items on an unopened/closed box must not complete S1 or S2/S3/S4.
- After S1, do not rely on YOLO always continuing to label `box_open`, because objects in the box can affect box-open detection.
- Do not use offline video for validation or runtime detection.

## Verification Results At Checkpoint

- `GET http://127.0.0.1/sop/detect` returned `HTTP/1.1 200 OK`.
- `GET http://127.0.0.1:18081/frames/latest.jpg` returned `HTTP/1.0 200 OK`, `Content-Type: image/jpeg`, size `103791` bytes.
- `python -m py_compile sop_system\ruoyi_hik_ws_bridge.py` passed.
- `npm.cmd run build:stage` passed with only existing asset-size warnings.
- Current RuoYi runtime was verified as `READY`, with all five steps `PENDING`, and both live stream URLs present.

## Before Any Future Change

1. Re-read this checkpoint.
2. Confirm the change does not touch detection core unless absolutely necessary.
3. If touching `ruoyi_hik_ws_bridge.py` or `index.vue`, recompute SHA256 and record a new checkpoint.
4. Verify at minimum:
   - `/sop/detect` opens.
   - `/frames/latest.jpg` returns a fresh real camera image.
   - Start/stop/reset work from the UI.
   - Correct S1-S5 action sequence passes.
   - Wrong sequence, especially skipping S1 and placing objects on a closed box, does not pass.
   - Step clips correspond to the actual step actions.
