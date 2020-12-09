@echo off
chcp 65001 >nul

if not exist "%USERPROFILE%\.pyvenvs\NarutoBattle\Scripts\python.exe" (  
	goto :ENV
)  

:RUN

call "%USERPROFILE%\.pyvenvs\NarutoBattle\Scripts\activate.bat"  
python ".\battle-gui.py"

exit

:ENV

echo Creating Environments for NarutoBalle.....

py -3.7 -m venv --prompt NarutoBattle "%USERPROFILE%\.pyvenvs\NarutoBattle"  
call "%USERPROFILE%\.pyvenvs\NarutoBattle\Scripts\activate.bat"  
python -m pip install -U pip  
pip install -U setuptools  
pip install -r env\requirements.txt

goto :RUN
