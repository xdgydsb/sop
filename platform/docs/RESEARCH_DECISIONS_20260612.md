# 工业动作检测外部调研与技术决策

日期：2026-06-12

## 1. 调研结论

参考视频对应的技术问题不是普通目标检测，也不是简单的整段视频动作分类，而是：

- 工业过程步骤识别（Procedure Step Recognition）。
- 在线动作分段（Online Action Segmentation）。
- 在线错误检测（Online Mistake Detection）。
- 手、物体、工具和工位状态的关系理解。
- 基于任务图的顺序、依赖和完成结果判定。

因此产品采用混合架构：

```text
实时视频
  -> 目标/手/工具检测
  -> 多目标跟踪与手物关系
  -> 状态与动作候选
  -> 在线时序模型（按需）
  -> SOP 任务图与规则运行时
  -> 工序判定、错误类型和证据
```

规则运行时负责可解释和安全的最终判定；学习型时序模型负责补充规则难以表达的复杂动作。两者不能互相替代。

## 2. 关键研究依据

### 2.1 “识别动作”不等于“确认工序成功”

IndustReal 将工业问题定义为过程步骤识别，强调工业现场更关心动作是否正确完成、顺序是否正确，而不只是动作是否出现。其数据同时包含遗漏和执行错误。

决策：

- 每步必须包含前置状态、动作过程、后置结果和步骤依赖。
- 模型输出动作类别只能成为证据，不能直接成为 `PASS`。
- 产品必须分别记录过程错误和执行结果错误。

来源：

- [IndustReal WACV 2024 paper](https://openaccess.thecvf.com/content/WACV2024/papers/Schoonbeek_IndustReal_A_Dataset_for_Procedure_Step_Recognition_Handling_Execution_Errors_WACV_2024_paper.pdf)
- [IndustReal repository](https://github.com/timschoonbeek/industreal)

### 2.2 实时检测必须采用在线/因果时序方法

ProTAS 的研究指出，将离线训练的动作分段模型直接用于在线推理会明显降低性能。在线系统还需要估计动作进度，并利用任务图中的步骤依赖。

决策：

- 生产模型不得使用未来帧。
- 训练、验证和线上推理必须使用一致的因果窗口。
- Runtime 保留明确的 `WAITING`、`IN_PROGRESS` 和完成保持阶段。
- 后续学习型时序模型必须输出动作候选、进度和不确定度。

来源：

- [Progress-Aware Online Action Segmentation, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf)

### 2.3 手与物体关系是错误理解的重要特征

CVPR 2024 的过程错误检测研究同时使用整体画面特征和活动物体关系特征，并明确指出手物交互有助于理解动作和错误。

决策：

- 首个视觉适配器必须输出持续目标 ID、物体轨迹和手物交互。
- 对“打开、放入、取出、装配”等动作，手物关系是触发条件之一。
- 仅依赖单帧物体标签的方案不进入生产链路。

来源：

- [Error Detection in Egocentric Procedural Task Videos, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_Error_Detection_in_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf)

### 2.4 未见过的错误不能只靠错误类别训练

PREGO 将在线错误检测设计为从正确流程学习正常顺序，把偏离正常依赖的行为视为候选错误，减少对所有错误样本都提前收集的依赖。

决策：

- 规则和任务图负责已知的强约束错误。
- 正确流程数据用于学习正常动作和正常顺序。
- 未知错误检测只能生成“异常/需复核”，不能未经解释直接判定具体责任。
- 错误样本用于验证和校准，但不假设能够穷举所有错误。

来源：

- [PREGO, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Flaborea_PREGO_Online_Mistake_Detection_in_PRocedural_EGOcentric_Videos_CVPR_2024_paper.pdf)

### 2.5 工业动作数据必须覆盖不同人员与模态

Assembly101 包含多视角、细粒度动作片段和大量手部姿态；InHARD 提供工业动作的 RGB、深度和骨架数据。这说明动作泛化需要跨人员、跨视角和多模态验证。

决策：

- 数据集按人员和采集批次划分训练/验证/测试，禁止相邻帧随机切分造成数据泄漏。
- 固定俯视相机优先使用目标轨迹和手物关系；全身骨架仅在工序确实依赖姿态时使用。
- 深度相机不是首期必需，但为遮挡严重和空间装配场景保留接口。

来源：

- [Assembly101 paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sener_Assembly101_A_Large-Scale_Multi-View_Video_Dataset_for_Understanding_Procedural_Activities_CVPR_2022_paper.pdf)
- [Assembly101 action-recognition repository](https://github.com/assembly-101/assembly101-action-recognition)
- [InHARD repository](https://github.com/vhavard/InHARD)

### 2.6 模型工具箱与生产视频管线应分离

MMAction2覆盖动作识别、定位、时空检测和骨架动作模型，适合作为实验和模型对比工具。DeepStream提供GPU优化的视频解码、批处理、推理和持久目标跟踪，适合NVIDIA环境下的多路生产视频管线。

决策：

- MMAction2只作为时序模型实验候选，不把整个框架嵌入领域运行时。
- 单海康USB相机阶段继续使用简单、可控的采集Worker。
- 当出现多路RTSP、边缘NVIDIA设备或GPU解码瓶颈时，再评估DeepStream。
- 生产部署通过统一Worker契约隔离OpenCV、DeepStream和模型框架。

来源：

- [MMAction2 repository](https://github.com/open-mmlab/mmaction2)
- [DeepStream architecture](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html)
- [DeepStream tracker](https://docs.nvidia.com/metropolis/deepstream-nvaie30/dev-guide/text/DS_plugin_gst-nvtracker.html)

## 3. 算法分级策略

### L1：状态转移规则

适用：开盒、关盒、物品由区域A进入区域B、数量改变。

算法：

- YOLO目标检测。
- ByteTrack、BoT-SORT、NvSORT或等价目标跟踪。
- ROI、包含、交并比、轨迹方向等几何关系。
- 连续帧和保持时间。
- SOP任务图。

这是首期默认方案，实时、可解释、数据量要求较低。

### L2：手物交互与姿态

适用：必须确认由工人拿取、按压、旋转、插入或使用工具。

算法：

- 手部检测或关键点。
- 人体姿态（按需）。
- 手-物距离、接触持续时间、共同运动和轨迹关联。
- 工具使用与安全区域规则。

首期包装流程需要手物交互，但不需要完整全身骨架。

### L3：学习型在线时序模型

适用：拧紧、插接、涂胶、检查等无法只靠前后状态可靠区分的动作。

候选：

- 轻量因果TCN或MS-TCN类模型。
- 因果动作进度模型。
- 基于骨架的ST-GCN/PoseC3D。
- RGB时序模型TSM/X3D/VideoMAE等的轻量化版本。

启用条件：

- L1/L2在真实测试中达不到目标指标。
- 已有足够的跨人员、跨批次标注数据。
- 实时延迟和算力预算经过验证。

### L4：异常与开放集错误

适用：无法预先列举的新错误。

策略：

- 从正确流程学习动作原型和依赖。
- 输出异常分数和人工复核建议。
- 不直接替代确定性错误规则。

## 4. 产品数据闭环

每个工序的数据应同时保存：

- 原始视频时间范围。
- 操作员、工位、产品型号、SOP版本和采集批次。
- 动作开始、结束和成功结果。
- 目标轨迹、手物关系和关键状态。
- 正确、错误、纠正和无法判断标签。
- 相机状态、遮挡、光照和推理版本。

错误分类至少包括：

- 遗漏。
- 增加或重复。
- 修改/错误执行。
- 错序。
- 动作中滑脱或失败。
- 发生错误后纠正。
- 系统中断或视觉不确定。

## 5. 验收指标

不能只用YOLO mAP或动作分类准确率验收产品。

生产指标：

- 正确步骤完成召回率。
- 错误动作拦截率。
- 错误通过率（False Accept Rate，重点指标）。
- 每步判定延迟P50/P95。
- 从错误发生到告警的延迟。
- 无法判断率和系统中断率。
- 步骤视频证据匹配率。
- 完整作业周期一次通过率。

模型研究指标：

- 帧级准确率。
- Segmental F1@10/25/50。
- Edit score，衡量过度分段和顺序质量。
- 跨人员、跨批次、跨光照测试结果。

## 6. 当前项目执行决定

立即执行：

1. 保留已实现的因果SOP Runtime。
2. 定义视觉适配器标准输出和可审计来源信息。
3. 实现固定俯视相机的目标轨迹、ROI和手物交互观察生成器。
4. 建立真实工位数据采集与标注规范。
5. 用L1/L2完成首个包装流程，再量化决定哪些步骤需要L3。

暂不执行：

- 不立即部署Video Transformer。
- 不立即切换DeepStream。
- 不把大模型视频理解放入实时主判定链。
- 不做无法通过真实相机验收的“自动训练”宣传功能。
