#!/bin/bash

# WildGS-SLAM 自动化安装脚本
# 适用于 Ubuntu 20.04 + CUDA 11.8 + RTX 4090

set -e  # 遇到错误立即退出

echo "=========================================="
echo "WildGS-SLAM 环境自动化搭建脚本"
echo "系统要求: Ubuntu 20.04, CUDA 11.8, RTX 4090"
echo "=========================================="

# 颜色输出函数
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_system() {
    log_info "检查系统环境..."
    
    # 检查Ubuntu版本
    if ! grep -q "Ubuntu 20.04" /etc/os-release; then
        log_warn "推荐使用 Ubuntu 20.04，当前系统可能存在兼容性问题"
    fi
    
    # 检查CUDA
    if ! command -v nvcc &> /dev/null; then
        log_error "未检测到CUDA，请先安装CUDA 11.8"
        exit 1
    fi
    
    cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    if [[ "$cuda_version" != "11.8" ]]; then
        log_warn "检测到CUDA版本为 $cuda_version，推荐使用11.8"
    fi
    
    # 检查GPU
    if ! nvidia-smi &> /dev/null; then
        log_error "未检测到NVIDIA GPU"
        exit 1
    fi
    
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    log_info "检测到GPU: $gpu_info"
    
    log_info "系统检查完成 ✓"
}

# 安装系统依赖
install_system_deps() {
    log_info "安装系统依赖..."
    
    sudo apt update
    sudo apt install -y \
        build-essential \
        cmake \
        ninja-build \
        git \
        wget \
        curl \
        python3-dev \
        python3-pip \
        python3-venv \
        libopencv-dev \
        libeigen3-dev \
        libceres-dev \
        libglew-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        freeglut3-dev
    
    log_info "系统依赖安装完成 ✓"
}

# 创建Python虚拟环境
create_venv() {
    log_info "创建Python虚拟环境..."
    
    # 安装Python 3.10 (如果需要)
    if ! python3.10 --version &> /dev/null; then
        log_info "安装Python 3.10..."
        sudo apt install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt update
        sudo apt install -y python3.10 python3.10-venv python3.10-dev
    fi
    
    # 创建虚拟环境
    python3.10 -m venv wildgs_slam_env
    source wildgs_slam_env/bin/activate
    
    # 升级pip
    pip install --upgrade pip setuptools wheel
    
    log_info "Python环境创建完成 ✓"
}

# 安装PyTorch
install_pytorch() {
    log_info "安装PyTorch和相关包..."
    
    source wildgs_slam_env/bin/activate
    
    # 安装PyTorch with CUDA 11.8
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118
    
    # 安装其他torch相关包
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html
    pip install xformers==0.0.22.post7+cu118 --index-url https://download.pytorch.org/whl/cu118
    
    # 验证PyTorch安装
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
    
    log_info "PyTorch安装完成 ✓"
}

# 克隆项目
clone_project() {
    log_info "克隆WildGS-SLAM项目..."
    
    if [ ! -d "WildGS-SLAM" ]; then
        git clone --recursive https://github.com/GradientSpaces/WildGS-SLAM.git
    else
        log_info "项目已存在，更新子模块..."
        cd WildGS-SLAM
        git submodule update --init --recursive
        cd ..
    fi
    
    log_info "项目克隆完成 ✓"
}

# 安装项目依赖
install_dependencies() {
    log_info "安装项目依赖..."
    
    source wildgs_slam_env/bin/activate
    cd WildGS-SLAM
    
    # 安装基础依赖（注意numpy版本）
    pip install numpy==1.26.3  # 重要：避免numpy 2.0兼容性问题
    pip install -r requirements.txt
    
    # 安装MMCV
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.1"
    
    log_info "基础依赖安装完成 ✓"
}

# 编译第三方库
compile_thirdparty() {
    log_info "编译第三方库..."
    
    source wildgs_slam_env/bin/activate
    cd WildGS-SLAM/thirdparty
    
    # 编译 lietorch
    log_info "编译 lietorch..."
    cd lietorch
    python setup.py install
    cd ..
    
    # 编译 diff-gaussian-rasterization
    log_info "编译 diff-gaussian-rasterization..."
    cd diff-gaussian-rasterization-w-pose
    pip install -e .
    cd ..
    
    # 编译 simple-knn
    log_info "编译 simple-knn..."
    cd simple-knn
    pip install -e .
    cd ..
    
    cd .. # 回到项目根目录
    
    log_info "第三方库编译完成 ✓"
}

# 下载预训练模型
download_models() {
    log_info "下载预训练模型..."
    
    cd WildGS-SLAM
    mkdir -p pretrained
    
    # 下载DROID模型
    if [ ! -f "pretrained/droid.pth" ]; then
        log_info "下载DROID模型..."
        wget -O pretrained/droid.pth \
            https://github.com/princeton-vl/DROID-SLAM/releases/download/v0.1/droid.pth
    else
        log_info "DROID模型已存在"
    fi
    
    # 预下载DINOv2模型
    log_info "预下载DINOv2模型..."
    source ../wildgs_slam_env/bin/activate
    python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"
    
    log_info "预训练模型下载完成 ✓"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    source wildgs_slam_env/bin/activate
    cd WildGS-SLAM
    
    # 运行Python验证脚本
    cat > verify_install.py << 'EOF'
import sys
import subprocess

def check_import(module_name, package_name=None):
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
            return True
        else:
            print("✗ GPU: CUDA not available")
            return False
    except Exception as e:
        print(f"✗ GPU Check failed: {e}")
        return False

print("========================================")
print("WildGS-SLAM 安装验证")
print("========================================")

# 检查Python版本
print(f"Python版本: {sys.version}")

# 检查核心依赖
modules_to_check = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("numpy", "NumPy"),
    ("cv2", "OpenCV"),
    ("mmcv", "MMCV"),
    ("lietorch", "LieTorch"),
    ("simple_knn", "Simple-KNN"),
    ("diff_gaussian_rasterization", "Diff-Gaussian-Rasterization"),
]

all_good = True
for module, name in modules_to_check:
    if not check_import(module, name):
        all_good = False

# 检查GPU
if not check_gpu():
    all_good = False

# 检查预训练模型
import os
if os.path.exists("pretrained/droid.pth"):
    print("✓ DROID模型文件")
else:
    print("✗ DROID模型文件缺失")
    all_good = False

print("========================================")
if all_good:
    print("🎉 安装验证成功！环境已就绪")
    print("\n激活环境命令:")
    print("source wildgs_slam_env/bin/activate")
    print("cd WildGS-SLAM")
else:
    print("❌ 安装验证失败，请检查错误信息")
    sys.exit(1)
EOF

    python verify_install.py
    rm verify_install.py
}

# 下载演示数据
download_demo_data() {
    log_info "下载演示数据..."
    
    cd WildGS-SLAM
    
    # 检查是否有下载脚本
    if [ -f "scripts_downloading/download_demo_data.sh" ]; then
        bash scripts_downloading/download_demo_data.sh
    else
        log_warn "演示数据下载脚本不存在，请手动下载测试数据"
    fi
    
    log_info "演示数据准备完成 ✓"
}

# 创建便捷脚本
create_convenience_scripts() {
    log_info "创建便捷脚本..."
    
    # 创建激活环境脚本
    cat > activate_wildgs.sh << 'EOF'
#!/bin/bash
# WildGS-SLAM环境激活脚本

echo "激活WildGS-SLAM环境..."
source wildgs_slam_env/bin/activate
cd WildGS-SLAM

echo "环境已激活！"
echo "当前目录: $(pwd)"
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "可用命令:"
echo "  python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo.yaml  # 运行演示"
echo "  python scripts_run/summarize_pose_eval.py                        # 查看结果"
echo ""
EOF
    chmod +x activate_wildgs.sh
    
    # 创建快速测试脚本
    cat > quick_test.sh << 'EOF'
#!/bin/bash
# WildGS-SLAM快速测试脚本

source wildgs_slam_env/bin/activate
cd WildGS-SLAM

echo "运行快速测试..."

# 检查是否有演示配置文件
if [ -f "configs/Dynamic/Wild_SLAM_Mocap/crowd_demo.yaml" ]; then
    echo "运行演示配置..."
    python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo.yaml
else
    echo "演示配置文件不存在，请检查数据集下载"
fi
EOF
    chmod +x quick_test.sh
    
    log_info "便捷脚本创建完成 ✓"
}

# 主函数
main() {
    log_info "开始WildGS-SLAM环境搭建..."
    
    check_system
    install_system_deps
    create_venv
    install_pytorch
    clone_project
    install_dependencies
    compile_thirdparty
    download_models
    verify_installation
    download_demo_data
    create_convenience_scripts
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}🎉 WildGS-SLAM环境搭建完成！${NC}"
    echo "=========================================="
    echo ""
    echo "快速开始："
    echo "1. 激活环境：  ./activate_wildgs.sh"
    echo "2. 运行测试：  ./quick_test.sh"
    echo ""
    echo "手动激活环境："
    echo "  source wildgs_slam_env/bin/activate"
    echo "  cd WildGS-SLAM"
    echo ""
    echo "运行演示："
    echo "  python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo.yaml"
    echo ""
    echo "查看结果："
    echo "  python scripts_run/summarize_pose_eval.py"
    echo ""
    echo "问题排查："
    echo "  查看日志文件或重新运行验证："
    echo "  source wildgs_slam_env/bin/activate && cd WildGS-SLAM && python verify_install.py"
    echo ""
}

# 捕获中断信号
trap 'log_error "安装被中断"; exit 1' INT

# 解析命令行参数
while getopts "h" opt; do
    case $opt in
        h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -h    显示帮助信息"
            echo ""
            echo "此脚本将自动安装WildGS-SLAM的完整环境。"
            echo "确保您的系统满足以下要求："
            echo "  - Ubuntu 20.04"
            echo "  - CUDA 11.8"
            echo "  - NVIDIA GPU (推荐RTX 4090)"
            echo "  - 32GB+ RAM"
            exit 0
            ;;
        \?)
            log_error "无效选项: -$OPTARG"
            exit 1
            ;;
    esac
done

# 运行主函数
main