@echo off
setlocal enabledelayedexpansion


:: Ejecuta python setup.py sdist
python setup.py sdist

:: Obtén el último archivo generado en el directorio dist
for /f "delims=" %%i in ('dir /b /o-d dist\*.tar.gz') do (
    set "latest_file=%%i"
    goto :break
)
:break

:: Sube el último archivo a PyPI
python -m twine upload -r pypi "dist\!latest_file!"

:: Finaliza el script
endlocal
