@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  SprintForm API  (FastAPI + Uvicorn)
echo  监听: http://127.0.0.1:8000  (文档: /docs)
echo  按 Ctrl+C 停止
echo ============================================
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
if errorlevel 1 (
  echo.
  echo [错误] 无法启动。请先在本目录执行依赖安装:
  echo   pip install -r requirements.txt
  echo.
  pause
)
