@echo off
cd /d "%~dp0"
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Khong chay duoc Streamlit. Hay chay lenh: python -m pip install -r requirements.txt
)
pause
