@echo off
echo ==================================================
echo  LMAFusion 启动脚本
echo ==================================================
echo.

call conda activate AGMFusion
if %errorlevel% neq 0 (
    echo [警告] 未能激活 AGMFusion 环境，尝试使用默认环境...
)

echo 当前 Python: 
python --version

echo.
echo 1. 安装依赖包
echo 2. 准备数据集
echo 3. 验证网络 (前向传播测试)
echo 4. 开始训练
echo 5. 推理测试
echo 6. 评估指标
echo.

set /p choice="请选择操作 (1-6): "

if "%choice%"=="1" (
    echo 安装依赖...
    pip install -r requirements.txt
    goto end
)

if "%choice%"=="2" (
    echo 准备无人机数据集...
    python prepare_data.py
    goto end
)

if "%choice%"=="3" (
    echo 验证网络前向传播...
    python net.py
    goto end
)

if "%choice%"=="4" (
    echo.
    set /p ep="训练轮数 (默认50): "
    set /p bs="批大小 (默认8): "
    if "%ep%"=="" set ep=50
    if "%bs%"=="" set bs=8
    echo 开始训练: epoch=%ep%, batch=%bs%
    python train.py --epoch %ep% --batch_size %bs%
    goto end
)

if "%choice%"=="5" (
    echo 推理测试...
    python test.py --model_path ./model/best.pth
    goto end
)

if "%choice%"=="6" (
    echo 评估指标...
    python evaluate.py
    goto end
)

:end
pause
