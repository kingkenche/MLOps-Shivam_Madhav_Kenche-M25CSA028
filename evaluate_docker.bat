@echo off
REM Quick Docker evaluation for pre-trained STL-10 model
echo [92m🐳 STL-10 Pre-trained Model Evaluation (Docker Desktop)[0m
echo.

REM Check if models exist
if not exist "models\best_model.pth" (
    echo [91m❌ Pre-trained model not found at models\best_model.pth[0m
    echo [93m⚠️  Please ensure your Colab-trained model is saved in the models folder[0m
    echo.
    echo Available model files:
    dir models\*.pth /b 2>nul
    if errorlevel 1 (
        echo   No .pth files found in models directory
    )
    pause
    exit /b 1
)

echo [92m✅ Pre-trained model found at models\best_model.pth[0m

REM Check if results exist
if exist "results\confusion_matrix.png" (
    echo [92m✅ Previous results found in results folder[0m
) else (
    echo [93m⚠️  No previous results found - will generate new evaluation[0m
)

echo.
echo [94mℹ️  Starting Docker evaluation of your pre-trained model...[0m
echo [94mℹ️  This will generate new evaluation results and visualizations[0m
echo.

REM Run the evaluation using the batch script
call docker-run.bat evaluate

echo.
echo [92m🎉 Evaluation Complete! Check the results folder for:[0m
echo   • confusion_matrix_evaluation.png
echo   • class_wise_accuracy_evaluation.png  
echo   • sample_predictions_evaluation.png
echo   • EVALUATION_SUMMARY.md

pause