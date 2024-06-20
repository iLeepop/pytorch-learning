# Learning Pytorch
well hello there

## 安装环境

### conda
https://www.anaconda.com/download
填写邮箱地址，下载地址会发送到邮箱里面
选择合适自己系统的版本
安装即可
#### windows
开始菜单会多出 `Anaconda Prompt`,点击启动
#### linux
运行 `conda init` 会初始化环境

#### 创建环境
运行 `conda create -n pytorch python=3.11.9`
运行 `conda activate pytorch` 激激活环境
`conda info`
`python --version`
`pip list`

### pytorch
https://pytorch.org/get-started/locally/
选择合适的选项，运行提供的命令即可

#### check
`python -c "import torch; print(torch.__version__); torch.cuda.is_avaliable()"`

### tensorboard
`pip install tensorboard`
启动服务`tensorboard --logdir=logs`

### opencv
`pip install opencv-python`

## 快速开始
`nn`就是神经网络`neural network`
分有很多层`layer`
`module`就是很多个层结合在一起，再通过各种优化得到最终模型


