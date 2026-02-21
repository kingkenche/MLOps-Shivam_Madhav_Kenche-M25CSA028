@echo off
REM Docker management script for STL-10 Classification (Windows)
REM Usage: docker-run.bat [build|run|stop|logs|shell|jupyter|gpu-test]

setlocal enabledelayedexpansion

set PROJECT_NAME=stl10-classification
set IMAGE_NAME=stl10-classifier
set CONTAINER_NAME=stl10-classification
set WANDB_KEY=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

echo [92m🐳 STL-10 Classification Docker Manager (Windows)[0m
echo.

REM Check if Docker is available
docker --version >nul 2>&1
if errorlevel 1 (
    echo [91m❌ Docker is not installed or not in PATH[0m
    exit /b 1
)

docker system info >nul 2>&1
if errorlevel 1 (
    echo [91m❌ Docker daemon is not running[0m
    exit /b 1
)

echo [92m✅ Docker is ready[0m

REM Check for NVIDIA Docker support
docker info | findstr /i "nvidia" >nul
if %errorlevel%==0 (
    echo [92m✅ NVIDIA Docker support detected[0m
    set RUNTIME_FLAG=--runtime=nvidia
    set GPU_ENV=-e NVIDIA_VISIBLE_DEVICES=all
) else (
    echo [93m⚠️  NVIDIA Docker support not found, will use CPU only[0m
    set RUNTIME_FLAG=
    set GPU_ENV=
)

set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=run

if "%COMMAND%"=="build" goto BUILD
if "%COMMAND%"=="run" goto RUN
if "%COMMAND%"=="evaluate" goto EVALUATE
if "%COMMAND%"=="jupyter" goto JUPYTER
if "%COMMAND%"=="shell" goto SHELL
if "%COMMAND%"=="stop" goto STOP
if "%COMMAND%"=="logs" goto LOGS
if "%COMMAND%"=="gpu-test" goto GPU_TEST
if "%COMMAND%"=="help" goto HELP
if "%COMMAND%"=="--help" goto HELP
if "%COMMAND%"=="-h" goto HELP

echo [91m❌ Unknown command: %COMMAND%[0m
goto HELP

:BUILD
echo [94mℹ️  Building Docker image...[0m
docker build -t %IMAGE_NAME% .
if errorlevel 1 (
    echo [91m❌ Failed to build Docker image[0m
    exit /b 1
)
echo [92m✅ Docker image built successfully[0m
goto END

:RUN
call :BUILD
echo [94mℹ️  Starting STL-10 training in Docker...[0m
docker run -it --rm ^
    %RUNTIME_FLAG% ^
    %GPU_ENV% ^
    -e WANDB_API_KEY=%WANDB_KEY% ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\results:/app/results" ^
    -v "%cd%\wandb:/app/wandb" ^
    -v "%USERPROFILE%\.cache\huggingface:/home/appuser/.cache/huggingface" ^
    -v "%USERPROFILE%\.cache\torch:/home/appuser/.cache/torch" ^
    --name %CONTAINER_NAME% ^
    %IMAGE_NAME%
goto END

:EVALUATE
call :BUILD
echo [94mℹ️  Evaluating pre-trained STL-10 model...[0m
echo [94mℹ️  Using existing model from Colab training[0m
docker run -it --rm ^
    %RUNTIME_FLAG% ^
    %GPU_ENV% ^
    -e WANDB_API_KEY=%WANDB_KEY% ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\results:/app/results" ^
    -v "%cd%\wandb:/app/wandb" ^
    -v "%USERPROFILE%\.cache\huggingface:/home/appuser/.cache/huggingface" ^
    -v "%USERPROFILE%\.cache\torch:/home/appuser/.cache/torch" ^
    --name %CONTAINER_NAME%-eval ^
    %IMAGE_NAME% ^
    python evaluate_pretrained.py
goto END

:JUPYTER
call :BUILD
echo [94mℹ️  Starting Jupyter notebook server...[0m
echo [94mℹ️  Jupyter notebook will be available at: http://localhost:8888[0m
docker run -it --rm ^
    %RUNTIME_FLAG% ^
    %GPU_ENV% ^
    -p 8888:8888 ^
    -e WANDB_API_KEY=%WANDB_KEY% ^
    -v "%cd%:/app" ^
    -v "%USERPROFILE%\.cache\huggingface:/home/appuser/.cache/huggingface" ^
    -v "%USERPROFILE%\.cache\torch:/home/appuser/.cache/torch" ^
    --name %CONTAINER_NAME%-jupyter ^
    %IMAGE_NAME% ^
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=""
goto END

:SHELL
call :BUILD
echo [94mℹ️  Starting interactive shell...[0m
docker run -it --rm ^
    %RUNTIME_FLAG% ^
    %GPU_ENV% ^
    -e WANDB_API_KEY=%WANDB_KEY% ^
    -v "%cd%:/app" ^
    -v "%USERPROFILE%\.cache\huggingface:/home/appuser/.cache/huggingface" ^
    -v "%USERPROFILE%\.cache\torch:/home/appuser/.cache/torch" ^
    --name %CONTAINER_NAME%-shell ^
    %IMAGE_NAME% ^
    /bin/bash
goto END

:STOP
echo [94mℹ️  Stopping all containers...[0m
for /f %%i in ('docker ps -q --filter "name=%PROJECT_NAME%" 2^>nul') do docker stop %%i
echo [92m✅ Containers stopped[0m
goto END

:LOGS
echo [94mℹ️  Showing container logs...[0m
docker logs -f %CONTAINER_NAME%
goto END

:GPU_TEST
call :BUILD
echo [94mℹ️  Testing GPU availability in container...[0m
docker run --rm ^
    %RUNTIME_FLAG% ^
    %GPU_ENV% ^
    %IMAGE_NAME% ^
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"
goto END

:HELP
echo.
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   build      Build the Docker image
echo   run        Run training (default)
echo   evaluate   Evaluate pre-trained model from Colab
echo   jupyter    Start Jupyter notebook server
echo   shell      Open interactive shell  
echo   stop       Stop all running containers
echo   logs       Show container logs
echo   gpu-test   Test GPU availability
echo   help       Show this help message
echo.
echo Examples:
echo   %0 build
echo   %0 run
echo   %0 jupyter
echo   %0 shell
goto END

:END
endlocal