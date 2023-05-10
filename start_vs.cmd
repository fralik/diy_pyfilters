@echo off
SETLOCAL ENABLEEXTENSIONS

IF "%VIRTUAL_ENV%" == "" (
    ECHO [31mError: Virtual environment is not activated, activate it and rerun the script [0m
    EXIT /b 1
)

set PYTHON_INCLUDE=
set PYBIND11_INCLUDE=

for /f "tokens=1,2" %%a in ('python -m pybind11 --includes') do (
    set PYTHON_INCLUDE=%%a
    set PYBIND11_INCLUDE=%%b
)

set PYTHON_INCLUDE=%PYTHON_INCLUDE:~2%
set PYBIND11_INCLUDE=%PYBIND11_INCLUDE:~2%
set PYTHON_LIBS=%PYTHON_INCLUDE%\..\libs
setx PYTHON_INCLUDE %PYTHON_INCLUDE%
setx PYBIND11_INCLUDE %PYBIND11_INCLUDE%
setx PYTHON_LIBS %PYTHON_LIBS%


echo PYTHON_INCLUDE=%PYTHON_INCLUDE%
echo PYBIND11_INCLUDE=%PYBIND11_INCLUDE%
echo PYTHON_LIBS=%PYTHON_LIBS%

rem Start VS Studio and exit the batch
start "" "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\devenv.exe"