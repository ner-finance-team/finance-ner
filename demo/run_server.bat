@echo off
chcp 65001 >nul
rem 금융 NER 데모 서버를 내 PC에서 켠다 (http://localhost:8000). 크롬 확장과 웹 페이지가 이 서버를 부른다.
rem   모델: demo\models\ 아래 폴더(또는 set NER_MODELS_DIR=D:\a;D:\b)
rem   Claude 비교(선택): 이 창에서 먼저  set ANTHROPIC_API_KEY=키   (파일에 저장하지 말 것)
setlocal
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [오류] 가상환경이 없습니다: %PY%
  echo        만들기:  python -m venv "%~dp0.venv"  ^&^&  "%PY%" -m pip install -r "%~dp0requirements.txt"
  pause ^& exit /b 1
)
echo 열기: http://localhost:8000/  (모델 목록: http://localhost:8000/api/models)
cd /d "%~dp0server"
"%PY%" -m uvicorn app:app --port 8000
pause
