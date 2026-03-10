@echo off
call D:\Users\Ricky-Li\anaconda3\Scripts\activate.bat AGMFusion
echo [INFO] Python: & python --version
echo [INFO] Checking numpy...
python -c "import numpy; print('numpy:', numpy.__version__)"
if errorlevel 1 goto install_deps

:start_train
echo.
echo [INFO] Starting training...
python train.py --epoch 50 --batch_size 8 --patch_size 128 --log_freq 20 --name LMAFusion_drone
goto end

:install_deps
echo [INFO] numpy/cv2 not available, trying conda install...
call D:\Users\Ricky-Li\anaconda3\Scripts\conda.exe install -n AGMFusion numpy opencv scikit-image -y
goto start_train

:end
pause
