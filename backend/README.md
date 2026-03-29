# SprintForm 后端

## 为什么会出现 `ECONNREFUSED 127.0.0.1:8000`

小程序请求的是 `http://localhost:8000`（即本机 8000 端口）。**必须先在本机启动下面的 API 服务**，否则就会「拒绝连接」。

## 第一次使用

```bash
cd backend
pip install -r requirements.txt
```

（建议使用虚拟环境：`python -m venv .venv`，再激活后安装。）

## 视频分析失败 / 一直显示分析失败

分析依赖 **MediaPipe 姿态模型文件**。若未下载，任务会在提取阶段失败，错误信息类似：

`MediaPipe model file not found ... Run backend/scripts/download_model.py`

**必须先下载模型（只需一次，或换电脑后重做）：**

```bash
cd backend
python scripts/download_model.py
```

默认使用 `full` 档位（可在 `.env` 切换 `POSE_MODEL_VARIANT`）：
- `backend/models/pose_landmarker_full.task`（默认）
- `backend/models/pose_landmarker_lite.task`
- `backend/models/pose_landmarker_heavy.task`

其他常见原因（**多数不是模型坏了**，而是规则/画质不满足）：

- **画质 QC 已改为分级**：竖屏、长时长、FPS 偏离不会直接硬失败；系统会尽量生成可分析片段并返回 warning。
- 可通过环境变量微调（在 `backend/.env`）：例如 `POSE_MODEL_VARIANT=full`、`QC_MIN_SEGMENT_DURATION_MS=600`、`QC_MAX_QC_GAP_FRAMES=10`。
- 视频里**全身入镜比例太低**、光线太差 → 检测不到足够多帧的姿态（或 QC 不通过）。
- **OpenCV 打不开**该编码（少见）→ 用 H.264 的 `.mp4` 重导出再试。
- 后端控制台会打 `analysis failed ... stage=pose_extract|qc|metrics` 日志，便于区分是哪一步挂的。
- 具体原因可看 `uploads/<video_id>.job.json` 里的 `"error"` 字段。
- 调试文件：
  - `uploads/<video_id>_pose_debug.json`
  - `uploads/<video_id>_qc_debug.json`

## 启动服务（Windows）

双击运行：

- **`start-server.bat`**

或在终端：

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动成功后，浏览器打开 <http://127.0.0.1:8000/docs> 应能看到接口文档。

## 微信开发者工具

- **详情 → 本地设置**：勾选 **不校验合法域名、web-view（业务域名）、TLS 版本以及 HTTPS 证书**（仅本地调试）。
- 保持 `utils/config.js` 里 `CURRENT_ENV = 'local'`，`apiBaseUrl` 指向本机。

## 真机调试

手机上的 `localhost` 不是电脑。请把 `apiBaseUrl` 改成电脑的 **局域网 IP**，例如 `http://192.168.1.10:8000`，并保证电脑防火墙放行 8000 端口。
