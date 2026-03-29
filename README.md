# TrekRun

WeChat mini-program + **FastAPI** backend for **side-view sprint video** upload and **async** pose-based analysis, aimed at high-school PE coaching workflows.

**中文简介：** 微信小程序上传侧视短跑视频，后端异步做姿态分析并输出步频、躯干前倾、摆臂、左右时相、技术稳定性等核心指标；面向体训/体考侧拍场景，不做伤病诊断或多人实时推理。

## Scope (MVP)

- Single athlete, **side-view** clips, standardized mid-sprint segment  
- Offline upload; analysis runs on the server  
- Five core metrics: **step rate**, **trunk lean (mean)**, **arm swing variability**, **left/right timing difference**, **technical stability score**

Out of scope for this repo: injury prediction, medical advice, multi-person or arbitrary-angle analysis, on-device real-time inference, rankings, or social features.

## Repository layout

| Path | Role |
|------|------|
| Repo root | WeChat mini-program (pages, components, `app.*`, `utils/`, etc.) |
| `backend/` | FastAPI app, video/pose pipeline, tests |

## Prerequisites

- **Python** 3.10+ (recommended)  
- **WeChat DevTools** for the mini-program  
- **MediaPipe** pose model file(s) under `backend/models/` (see below)

Optional: MySQL, object storage, and a task queue (Celery/RQ) depending on your deployment; local dev can run with defaults in `.env.example`.

## Backend (local)

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env
python scripts/download_model.py
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API docs: <http://127.0.0.1:8000/docs>  
- Windows shortcut: `backend/start-server.bat`

Fill `backend/.env` from `.env.example`. Never commit real secrets (`.env` is gitignored).

## Mini-program (local debugging)

1. Open the project root in **WeChat DevTools**.  
2. **Details → Local settings**: enable **skip TLS / legal domain checks** (dev only).  
3. In `utils/config.js`, set `CURRENT_ENV` to `'local'` so the app calls your machine (`devtoolsApiBaseUrl`, e.g. `http://127.0.0.1:8000`).  
4. On a **physical device**, `localhost` is wrong; use the PC’s LAN IP and open the firewall for port `8000`.

For **mock mode** without a backend, use `CURRENT_ENV = 'dev'` and `useMock: true` in that env block.

## Troubleshooting

See **[backend/README.md](backend/README.md)** for:

- `ECONNREFUSED` / backend not running  
- Analysis failures, QC, and MediaPipe model paths  
- Where to read `uploads/<video_id>.job.json` errors

## Tests (backend)

```bash
cd backend
.venv\Scripts\activate   # or source .venv/bin/activate
pytest
```

## License

See [LICENSE](LICENSE) in the repository root.
