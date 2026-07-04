#!/bin/bash
# CUDA "unknown error" 修复脚本
# 使用方法: bash fix_cuda.sh

echo "================================================"
echo "CUDA Unknown Error 修复脚本"
echo "================================================"

# 方案1: 重载 nvidia_uvm 模块
echo ""
echo "[方案1] 重载 nvidia_uvm 内核模块..."
sudo rmmod nvidia_uvm 2>/dev/null
if [ $? -eq 0 ]; then
    sudo modprobe nvidia_uvm
    echo "nvidia_uvm 模块已重载"
else
    echo "nvidia_uvm 正在使用中，尝试方案2..."
    
    # 方案2: 重载所有 nvidia 模块
    echo ""
    echo "[方案2] 重载所有 NVIDIA 内核模块（会短暂黑屏）..."
    sudo systemctl stop gdm3 2>/dev/null || sudo systemctl stop lightdm 2>/dev/null
    sleep 2
    sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null
    sudo modprobe nvidia
    sudo modprobe nvidia_uvm
    sudo systemctl start gdm3 2>/dev/null || sudo systemctl start lightdm 2>/dev/null
    echo "所有 NVIDIA 模块已重载"
fi

echo ""
echo "验证修复结果..."
/home/daylight/anaconda3/envs/DjangoProject3/bin/python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA 版本:', torch.version.cuda)
    print('修复成功!')
else:
    print('修复失败，请尝试重启系统')
"

echo ""
echo "================================================"