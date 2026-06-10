@echo off
setlocal

python "%~dp0cogsb-gui-launch.py" %*

set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
