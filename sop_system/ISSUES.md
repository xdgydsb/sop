# SOP系统 — 当前问题梳理

## 项目概述

实时SOP检测系统，5步装配流程：
- S1: 打开纸盒 → S2: 放入耳机盒 → S3: 放入充电插头 → S4: 放入绿色袋子 → S5: 关闭纸盒

摄像头(海康威视) → YOLO(CUDA/RTX4090) → 多层级稳定器 → 事件检测 → FSM状态机 → 客户端UI

## 文件结构

```
sop_system/
├── engine/                          # 核心检测引擎（在服务器上运行）
│   ├── yolo_detector.py             # YOLO目标检测 (CUDA)
│   ├── hand_detector.py             # MediaPipe手部检测
│   ├── detection_stabilizer.py      # 检测稳定器 - EMA置信度+空间一致性
│   ├── box_state_stabilizer.py      # 盒子状态稳定器 - OPEN/CLOSED互斥
│   ├── object_state_tracker.py      # 物体状态追踪器 - 物体放入检测
│   ├── event_detector.py            # 事件检测器 - 输出5个标准事件
│   ├── sop_fsm.py                   # SOP有限状态机 - 序列约束
│   ├── action_segmenter.py          # 动作分段器 - 动作阶段识别
│   ├── physical_state.py            # 物理状态引擎 (旧版，与object_state_tracker功能重叠)
│   ├── temporal_predictor_v2.py     # 时序预测器 TCN+BiGRU (辅助用)
│   ├── temporal_lstm.py             # 旧版LSTM预测器
│   ├── fusion.py                    # 融合模块
│   ├── hmm_filter.py                # HMM滤波器
│   └── pink_marker_detector.py      # 粉色标记检测器
├── server_inference/
│   └── inference_server.py          # WebSocket服务器 (主线集成所有模块)
├── local_client/
│   └── hik_stream_client.py         # Windows客户端 (海康相机MVS SDK + UI)
├── tools/                           # 工具脚本
├── training/                        # 训练脚本
├── camera/                          # 相机驱动
├── utils/                           # 工具函数
├── models/                          # 模型文件
├── data/                            # 数据
├── reports/                         # 报告输出
└── config.py                        # 配置文件
```

## 核心数据流

```
Camera JPEG → YOLO detect → DetectionStabilizer(EMA) → BoxStateStabilizer(OPEN/CLOSED)
                                                       → ObjectStateTracker(per-object state machine)
                       → EventDetector(expected_event gate + physical conditions)
                       → SOPStateMachine.validate_event() → step advancement → UI
                       → TemporalPredictorV2 (仅辅助，不驱动FSM)
```

---

## 🔴 问题1: 检测框闪烁/乱飘 (最关键)

### 现象
物体检测框在画面中不断跳动、飘移、消失又出现

### 可能根因

**A. YOLO检测本身不稳定**
- YOLO对同一物体相邻帧的bbox可能差异很大（位置抖动、尺寸变化）
- 低置信度时物体的bbox不准确
- box_open/box_closed是同一个物理盒子但被检测为两个不同类，它们的bbox可能不重叠

**B. DetectionStabilizer的EMA平滑力度不够**
- 配置: ema_alpha=0.35 (个别0.30-0.40)
- 这意味着新检测权重35%，历史权重65% — 对于快速移动可能过于敏感
- 空间一致性闸门(IoU<0.15)可能太宽松，导致不同位置的false positive被连接成同一物体

**C. BoxStateStabilizer与DetectionStabilizer各自独立**
- DetectionStabilizer对box_open和box_closed分别做EMA
- BoxStateStabilizer再用自己的"higher-confidence-wins"逻辑重新判断
- 两者可能对同一帧的box状态给出不同结论
- 客户端显示来自DetectionStabilizer.get_display_detections()，但box互斥逻辑在BoxStateStabilizer
- **双重稳定器互相冲突**：DetectionStabilizer可能确认box_open，BoxStateStabilizer判定CLOSED，显示层再强行删除box_open → 框闪烁

**D. anti-flicker grace period (3帧)不够**
- 丢失后保留3帧，对20fps=150ms — 人眼可感知的闪烁
- 如果YOLO连续4帧检测不到，框就消失然后下次出现时位置完全不同

**E. charger fallback检测产生不同位置的bbox**
- YOLO找不到charger时，white_guided_search在白色区域重新搜索
- fallback的bbox与YOLO原始bbox位置可能完全不同
- 两个来源的bbox交替出现 → 框跳来跳去

### 涉及文件
- [engine/detection_stabilizer.py](engine/detection_stabilizer.py)
- [engine/box_state_stabilizer.py](engine/box_state_stabilizer.py)
- [engine/yolo_detector.py](engine/yolo_detector.py)
- [server_inference/inference_server.py](server_inference/inference_server.py) (line 986-1031 显示处理)

---

## 🔴 问题2: 动作识别完全不正确

### 现象
S2/S3/S4的动作无法被正确识别，即使用户做了对应动作，系统也不触发事件

### 可能根因

**A. ObjectStateTracker的init_roi=None逻辑**
- 当init_roi=None时，"初始位置"被定义为"盒子外部"
- 物体必须在盒子外部被看到≥outside_box_min(5)帧才算init_seen
- 然后物体必须离开初始位置(LEFT_INIT)，再进入盒子(VISIBLE_IN_BOX)，稳定N帧后才算READY
- **问题**: 如果物体一开始就在盒子附近(或盒子内部)，它永远不会经过"盒子外部"阶段，永远无法触发事件

**B. EventDetector的expected_event门控过于严格**
- 每帧只检查expected_event这一个事件
- 如果expected_event没有正确推进（卡在某个事件上），后续事件永远不会被检查
- expected_event的推进依赖事件被成功emit并accept，任何一个环节出错就卡死

**C. "手部遮挡=物体消失"**
- YOLO检测不到被手遮挡的物体 → detected=False
- ObjectStateTracker中物体进入OCCLUDED状态
- 如果手一直拿着物体(遮挡)，物体永远不会进入VISIBLE_IN_BOX状态
- **矛盾**: 用户放东西时手必然遮挡物体

**D. EventDetector cooldown (60帧≈3秒)过长**
- 事件发射后要等60帧才能检测下一个事件
- 用户快速操作(S2做完立刻S3)时，系统还在冷却中

**E. 多个状态机状态不一致**
- ObjectStateTracker跟踪per-object状态
- PhysicalStateEngine也跟踪per-object状态（旧版）
- EventDetector只看ObjectStateTracker
- 但SOPServer初始化时同时创建了PhysicalStateEngine和ObjectStateTracker
- **两者可能对同一物体给出不同结论**

**F. TemporalPredictor不给力**
- step_probs对当前步骤的预测可能不准确
- temporal_conf_boost最多只加0.10，帮助有限
- 如果event_confidence本来就不高，boost后仍不够

### 涉及文件
- [engine/object_state_tracker.py](engine/object_state_tracker.py)
- [engine/event_detector.py](engine/event_detector.py)
- [engine/physical_state.py](engine/physical_state.py)

---

## 🔴 问题3: S3/S4自动触发（乱序）

### 现象
S2完成后，用户还没做S3，S3和S4就自动完成了

### 可能根因

**A. ObjectStateTracker中init_roi=None逻辑缺陷**
- 当物体一开始就在盒子内或盒子附近，`in_init = not in_box`
- 如果物体恰好在盒子边缘被检测到，in_init和in_box频繁切换
- 可能物体从未真正在外面，但系统认为它经过了完整状态链

**B. stable_box_frames累积条件太宽松**
- 当前`stable_in_box_min=3` — 物体中心在盒子内3帧就算稳定
- 如果YOLO检测不稳定，物体可能"闪现"在盒子内3帧

**C. was_outside_box 条件可能被绕过**
- 如果物体最初几帧没被检测到，后来直接在盒子内被检测到
- was_outside_box可能为False，但left_init_roi可以为True(通过init_roi=None逻辑)

### 涉及文件
- [engine/object_state_tracker.py](engine/object_state_tracker.py) (line 87-252)
- [engine/event_detector.py](engine/event_detector.py) (line 234-248)

---

## 🔴 问题4: S5关盒无法完成

### 现象
S1-S4都完成后，用户关闭盒子，系统不触发box_closed事件

### 可能根因

**A. BoxStateStabilizer需要closed_high_thr ≥ 0.65 且连续5帧**
- 关盒动作很快(<0.5秒)，可能不到5帧或第3-4帧时置信度掉落
- 之前改用confirm_frames=5但可能对关盒动作太严格

**B. box_closed mapped to COMPLETE跳过了S5_CLOSE**
- 修改后box_closed直接映射到COMPLETE
- 但VALID_TRANSITIONS中S4_BAG→COMPLETE是合法的
- 问题可能出在EventDetector的_check_box_closed要求S2/S3/S4全部done
- **如果S2/S3/S4中任何一个没有被正确emit到_accepted_events，box_closed永远不会触发**

**C. 关盒时box_open和box_closed置信度都很低**
- 手在关盒时遮挡盒子，YOLO既看不到open也看不到closed
- BoxStateStabilizer进入TRANSITION或UNKNOWN状态
- EventDetector的_check_box_closed要求box_state==CLOSED且stable≥3帧

### 涉及文件
- [engine/event_detector.py](engine/event_detector.py) (line 250-289)
- [engine/sop_fsm.py](engine/sop_fsm.py) (line 37-44)
- [engine/box_state_stabilizer.py](engine/box_state_stabilizer.py)

---

## 🔴 问题5: 已完成步骤回退

### 现象
已显示绿色的步骤(S1/S2/S3/S4)又变回未完成状态

### 可能根因

**A. _max_fsm_path 只缓存路径长度，不缓存具体步骤**
- `_max_fsm_path = [1, 2]` 不包含步骤ID 3
- 如果_accepted_events被意外清空或修改
- _max_fsm_path虽不会缩小但可能不完整

**B. reset时_accepted_events被清空**
- stop()/reset()时清空_accepted_events和_max_fsm_path
- 如果客户端意外触发reset（如网络重连），步骤全丢

**C. FSM状态机回退**
- FSM的validate_event可以接受ERRR状态然后auto-recover
- auto-recover会回退到前一步骤
- 回退后fsm_path从FSM.current_step计算可能变小

### 涉及文件
- [server_inference/inference_server.py](server_inference/inference_server.py) (line 744, 1253-1272)
- [engine/sop_fsm.py](engine/sop_fsm.py) (line 145-155)

---

## 🔴 问题6: 架构过度复杂 — 模块职责重叠

### 现象
同一功能由多个模块重复实现，互相可能冲突

### 重叠点

| 功能 | 模块1 | 模块2 | 冲突风险 |
|------|-------|-------|---------|
| 检测稳定化 | DetectionStabilizer(EMA) | YOLODetector(bbox_ema) | 双重EMA |
| 盒子状态 | BoxStateStabilizer | DetectionStabilizer(box_open/closed) | 状态不一致 |
| 物体放置追踪 | ObjectStateTracker | PhysicalStateEngine | 两个状态机 |
| 事件检测 | EventDetector | 旧代码中的直接FSM调用 | 两条路径 |
| 时序预测 | TemporalPredictorV2 | TemporalPredictor(TFLite旧版) | 两个模型 |

**PhysicalStateEngine** 在inference_server.py line 694被初始化但似乎没有在process()中被实际使用（使用的是ObjectStateTracker），但它占用了初始化时间和内存。

---

## 建议简化方向

1. **去掉DetectionStabilizer对box_open/box_closed的状态判断** — 盒子状态完全交给BoxStateStabilizer，DetectionStabilizer只负责非box类的物体
2. **合并ObjectStateTracker和PhysicalStateEngine** — 只保留一个per-object状态机
3. **YOLO的bbox_ema和DetectionStabilizer的EMA二选一** — 不要双重平滑
4. **降低confirm_frames** — 对于关盒这种快速动作，2-3帧就够了
5. **增加init_roi显式指定** — 不用None逻辑，而是根据实际场景标定每个物体的初始区域
6. **事件cool down缩短到15帧** — 3秒太长了
7. **手部遮挡处理** — 当hand_near=True时，即使detected=False也保持物体visible状态更长时间

---

## 服务器信息

- 服务器: 192.168.31.19:8765 (Linux, RTX 4090)
- Python路径: ~/miniconda3/envs/three_env/bin/python
- 启动命令: `cd /home/zhaowei/shabi/sop_system && ~/miniconda3/envs/three_env/bin/python server_inference/inference_server.py --stage 3 --imgsz 640`
- 客户端: `cd d:\shabi\sop_system && python local_client/hik_stream_client.py`
