# RuoYi Runtime Bridge

## 作用

- 把真实检测运行态推送到 RuoYi 的 `sop/detect` 页面
- 提供实时 MJPEG 预览流
- 为每个步骤保存截图和 MP4 片段
- 把真实步骤事件同步到 `/sop/runtime/sync`

## 数据库升级

按顺序执行：

1. `SOP前后端/sql/sop_visual_detection.sql`
2. `SOP前后端/sql/sop_runtime_upgrade.sql`
3. 如果菜单缺失，再执行 `SOP前后端/sql/sop_detect_menu.sql`

## 启动 RuoYi

在 `SOP前后端` 目录：

```powershell
mvn -pl ruoyi-admin -am -DskipTests package
```

然后按现有方式启动 `ruoyi-admin`。

## 启动桥接

### 推荐方案

本地电脑接海康 USB 工业相机，服务器跑 `stage3` 推理：

```powershell
py -3 ruoyi_hik_ws_bridge.py `
  --server 192.168.31.19 `
  --port 8765 `
  --product-id 100 `
  --sop-id 100 `
  --ruoyi-base-url http://192.168.31.19:8080 `
  --public-host <本机局域网IP>
```

说明：

- `--public-host` 必须是打开 RuoYi 页面那台浏览器能访问到的本机 IP
- `--ruoyi-base-url` 指向实际运行中的 RuoYi 后端地址
- 不指定 `--task-code` 时，页面点击“开始检测/重置检测”会自动创建新任务

### 备选方案

如果确实需要本地推理，可用：

```powershell
py -3 ruoyi_runtime_bridge.py `
  --product-id 100 `
  --sop-id 100 `
  --ruoyi-base-url http://127.0.0.1:8080 `
  --public-host 127.0.0.1 `
  --camera webcam
```

## 页面说明

打开 `SOP检测页面` 后：

- 主画面读取 `previewStreamUrl`，应为实时视频流
- 右侧步骤状态来自真实 runtime 数据
- 下方步骤卡片展示对应截图和片段
- “开始检测 / 停止检测 / 重置检测” 通过 runtime session 控制 `stage3`

## 输出目录

- 预览与片段默认保存在 `sop_system/runtime_outputs_ws/`
- 最新截图在 `frames/latest.jpg`
- 步骤片段在 `clips/`
