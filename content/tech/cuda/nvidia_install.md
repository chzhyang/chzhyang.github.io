---
title: "NVIDIA Driver and CUDA Installation"
date: 2024-03-18T09:47:15Z
draft: false
description: ""
tags: ["GPU"]
# layout: "simple"
showDate: true
---

## Pre-requirements

OS: ubuntu 22.04, x86-64, kernel(6.5.0-35-generic)
NVIDIA GPU: GeForce RTX 3090, 24GB

搜索和下载合适版本的 driver 和 cuda，并给下载后的 runfiles 加上 执行权限
- [NVIDIA driver 下载](https://www.nvidia.cn/drivers/lookup/)， `NVIDIA-Linux-x86_64-550.90.07.run`
- [CUDA 下载](https://developer.nvidia.com/cuda-downloads)， `wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42.02_linux.run`

禁用 Nouveau（开源版 NVIDIA Driver）

    修改blacklist.conf，在文件末尾加上

    ```
    sudo vim /etc/modprobe.d/blacklist.conf
    ```

    ```
    blacklist nouveau
    options nouveau modeset=0
    ```

    重新生成 kernel initramfs

    ```
    sudo dracut --force
    sudo grub2-mkconfig -o /boot/grub2/grub.cfg
    ```

    Reboot 后检查，没有信息返回表示成功禁用

    ```
    lsmod | grep nouveau
    ```

## Install Driver

卸载旧版本驱动

```
sudo apt-get remove --purge nvidia*
```

### 安装
```
sudo ./NVIDIA-Linux-x86_64-550.90.07.run -no-x-check -no-nouveau-check -no-opengl-files
```
- -no-x-check：安装驱动时关闭X服务
- -no-nouveau-check：安装驱动时禁用nouveau
- -no-opengl-files：只安装驱动文件，不安装OpenGL文件

安装过程中的选项：
```
The distribution-provided pre-install script failed! Are you sure you want to continue? 
选择 yes 
Would you like to register the kernel module souces with DKMS? This will allow DKMS to automatically build a new module, if you install a different kernel later?  
选择 No 
Would you like to run the nvidia-xconfigutility to automatically update your x configuration so that the NVIDIA x driver will be used when you restart x? Any pre-existing x confile will be backed up.  
选择 Yes
```



### 首次安装Driver失败
原因是ubuntu22.04的kernel(6.5.0-35-generic)自带的 gcc-11，不符合 driver 的版本要求，参考 [link1](https://forums.developer.nvidia.com/t/issues-with-the-driver-installation-after-updating-to-the-6-5-0-35-generic-ubuntu-kernel/294142/3) 和 [link2](https://forums.developer.nvidia.com/t/driver-install-fails-with-the-error-an-error-occurred-while-performing-the-step-building-kernel-modules-see-var-log-nvidia-installer-log/280385)

安装Driver失败时，`/var/log/nvidia-installer.log`中的部分提示信息

```
make[1]: Entering directory '/usr/src/linux-headers-6.5.0-35-generic'
   warning: the compiler differs from the one used to build the kernel
     The kernel was built by: x86_64-linux-gnu-gcc-12 (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
     You are using:           cc (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
     SYMLINK /tmp/selfgz67285/NVIDIA-Linux-x86_64-550.90.07/kernel/nvidia/nv-kernel.o
     SYMLINK /tmp/selfgz67285/NVIDIA-Linux-x86_64-550.90.07/kernel/nvidia-modeset/nv-modeset-kernel.o
```

解决方法：安装 gcc-12 和 g++-12
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-12 g+±12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 --slave /usr/bin/g++ g++ /usr/bin/g+±12
sudo update-alternatives --config gcc
```

安装 gcc-12 时又遇到了 dpkg 报错， 且使用 `sudo apt-get clean` 和 `sudo apt-get autoremove` 无法解决

```
Errors were encountered while processing: 
/var/cache/apt/archives/libglvnd0_1.7.0-2101~22.04_i386.deb 
/var/cache/apt/archives/libglx0_1.7.0-2101~22.04_i386.deb 
/var/cache/apt/archives/libgl1_1.7.0-2101~22.04_i386.deb
```

解决方法是强制重写报错的这几个package，参考 [link](https://askubuntu.com/questions/141370/how-to-fix-a-broken-package-when-apt-get-install-f-does-not-work)

```
sudo dpkg -i --force-overwrite path-to-the-deb-file
sudo apt-get autoremove
sudo apt-get install -f
```

### 检查是否安装成功
```
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:B3:00.0 Off |                  N/A |
| 39%   49C    P0             99W /  370W |       1MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

```

## Install CUDA

```
$ sudo sh ./cuda_12.5.0_555.42.02_linux.run 
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-12.5/

Please make sure that
 -   PATH includes /usr/local/cuda-12.5/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.5/lib64, or, add /usr/local/cuda-12.5/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-12.5/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 555.00 is required for CUDA 12.5 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log

```

cat /usr/local/cuda/version.txt
cat /usr/local/cuda/version.json

```
# nv
export PATH=/usr/local/cuda-12.5/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

```
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Apr_17_19:19:55_PDT_2024
Cuda compilation tools, release 12.5, V12.5.40
Build cuda_12.5.r12.5/compiler.34177558_0
```

## Try CUDA Samples

Download and build [CUDA Samples](https://github.com/NVIDIA/cuda-samples/tree/master)
```
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
make
```
运行 CUDA kernel
```
$ ./cuda-samples/bin/x86_64/linux/release/matrixMul

[Matrix Multiply Using CUDA] - Starting...
GPU Device 0: "Ampere" with compute capability 8.6

MatrixA(320,320), MatrixB(640,320)
Computing result using CUDA Kernel...
done
Performance= 2291.55 GFlop/s, Time= 0.057 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```
## Reference

- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
- [Issues with the driver installation after updating to the 6.5.0-35-generic Ubuntu kernel](https://forums.developer.nvidia.com/t/issues-with-the-driver-installation-after-updating-to-the-6-5-0-35-generic-ubuntu-kernel/294142/3)
- [Driver install fails with the error \[An error occurred while performing the step: “building kernel modules”. See /var/log/nvidia-installer.log…\]](https://forums.developer.nvidia.com/t/driver-install-fails-with-the-error-an-error-occurred-while-performing-the-step-building-kernel-modules-see-var-log-nvidia-installer-log/280385)
- [How to fix a broken package, when "apt-get install -f" does not work?](https://askubuntu.com/questions/141370/how-to-fix-a-broken-package-when-apt-get-install-f-does-not-work)
- [Ubuntu20.04系统，3090显卡，安装驱动、CUDA、cuDNN的步骤](https://segmentfault.com/a/1190000040322236)