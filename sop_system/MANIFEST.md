# SOP System — 文件清单
# Updated: 2026-05-20 (v2_90 基线冻结, 阶段3回放验证中)
# Source: zhaowei@192.168.31.19:/home/zhaowei/shabi

## 当前版本: v2_90_tcn_bigru_center_T48 (FROZEN 2026-05-20)
- 输入90维, T=48, center-frame, 6类输出
- 离线: 97.15% step acc, OK通过率 98.9%, WR检出率 100%
- 详见: reports/temporal_v2_90/VERSION_SUMMARY.md

## engine/ — 核心引擎
- yolo_detector.py       — YOLO+ByteTrack 目标检测追踪 [稳定,勿改]
- hand_detector.py       — MediaPipe 手部21关键点 [稳定,勿改]
- sop_fsm.py             — SOP有限状态机 S1→S2→S3→S4→S5 [稳定,勿改]
- temporal_predictor_v2.py — ★ v2_90 实时推理包装器 [Phase2, 增量模式需修复]
- physical_state.py      — 物理状态引擎 (盒子/物体放置判定)
- temporal_lstm.py       — 旧版BiGRU (130维) [已废弃]
- fusion.py              — 旧版融合决策 [已废弃]
- hmm_filter.py          — HMM滤波器 [未使用]

## tools/ — 工具脚本
- extract_features_v2_90.py       — 离线提取90维特征 (15fps)
- build_frame_labels_v2.py        — segments→逐帧标签
- build_temporal_windows_v2.py    — 构建训练窗口 (T=48, stride=8)
- audit_v2_90_integrity.py        — ★ v2_90可信性审查 (5项全通过)
- replay_realtime_pipeline_v2.py  — ★ 原始视频回放测试 [Phase3, 待修复]
- convert_manual_npy_to_segments.py
- config_v2.py

## models/ — 模型
- yolo_final_v1.pt                  (50MB) — YOLO检测 mAP50=0.95
- temporal/v2_90_tcn_bigru/checkpoints/best.pt (11MB) — ★ 当前模型
- best_sequence_v5.pt               (12MB) — 旧版BiGRU 130维 [废弃]
- sop_step_detector.pt              (16MB) — 旧版步骤检测器 [废弃]
- yolo_v6_best.pt                   (50MB) — 对比保留

## data/ — 数据 (完整数据在服务器上)
- test_videos/ok/  (10个avi), wr/ (10个avi) — 本地测试用
- features/v2_90/  (完整245文件在服务器, 本地仅ok_1)
- labels/frame_labels_v2/ (完整244文件在服务器, 本地仅ok_1)
- datasets/temporal_v2_90_T48_S8/ (完整5942窗口在服务器)
- annotations/segment_annotations_v2.csv — 人工标注segment

## reports/
- STATUS_2026-05-20.md — ★ 完整状态报告 (当前工作进度)
- temporal_v2_90/VERSION_SUMMARY.md — v2_90 冻结基线

## 当前状态 (2026-05-20)
- Phase 0-2: ✅ 完成
- Phase 3 (回放验证): ⚠️ 卡住 — 预测器增量/批量不一致导致OK通过率只有75%
- Phase 4-6: ⏳ 待定
- 详细分析见: reports/STATUS_2026-05-20.md
