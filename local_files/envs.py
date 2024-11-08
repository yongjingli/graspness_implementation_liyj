# ModuleNotFoundError: No module named 'distutils.msvccompiler'
# downgrade the Python version to 3.7

# subprocess.CalledProcessError: Command '['ninja', '-v', '-j', '12']' returned non-zero exit status 1.

# 最好使用cuda-11.1进行安装
# 安装教程#
# https://github.com/NVIDIA/MinkowskiEngine#anaconda
# conda create -n py3-mink python=3.8
# conda activate py3-mink
#
# conda install openblas-devel -c anaconda
# conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
#
# # Install MinkowskiEngine
#
# # Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 11.X
# # export CUDA_HOME=/usr/local/cuda-11.1
# pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"


# 预训练模型下载
# https://github.com/user432/gamma
# Download the model from this link and place it in graspness_implementation/data/

# RuntimeError: "floor" "_vml_cpu" not implemented for 'Int'
# 参考https://blog.csdn.net/qq_44940689/article/details/138618257
# .../miniconda3/envs/grasp/lib/python3.8/site-packages/MinkowskiEngine-0.5.4-py3.8-linux-x86_64.egg/MinkowskiEngine/utils/quantization.py
# return torch.floor(array)
# # 改为：
# return torch.floor(array.float())

# 安装pytorch3d 从源码安装
# git clone https://github.com/facebookresearch/pytorch3d
# cd pytorch3d && pip install -e .  # 会出现很多版本的问题

# https://github.com/facebookresearch/pytorch3d/discussions/1752
# 根据自己的版本，根据最接近的版本进行安装 pytorch 1.9 替换为1.11   cuda 11.1替换为11.3  用别人编译好的包
# 安装的是11.1的cuda版本，如果cuda不对应的话还是不行，所以cuda的版本非常的关键
# pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.5+pt1.11.0cu113

# RuntimeError: CUDA error: an illegal memory access was encountered
# CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
# 应该是mask为空时出现的,没有可以输入的objectness点
# 设置一个假的点
# fake true
# if torch.sum(graspable_mask) == 0:
#     graspable_mask[:, 0] = True