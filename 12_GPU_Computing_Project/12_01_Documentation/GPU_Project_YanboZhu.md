

# 1 Setup


## 1.1 install dependency


```
sudo apt update
sudo apt install -y build-essential cmake git unzip pkg-config
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install -y libxvidcore-dev libx264-dev libx265-dev
sudo apt install -y libgtk-3-dev libcanberra-gtk3-dev
sudo apt install -y libatlas-base-dev gfortran
sudo apt install -y python3-dev python3-pip


```

## 1.2 Install Opencv

```bash
mkdir opencv
cd opencv
git clone https://github.com/opencv/opencv.git
git checkout 4.13.0

mkdir opencv_contrib
cd opencv_contrib
git clone https://github.com/opencv/opencv_contrib.git
git checkout 4.13.0

```


### 1.2.1 compile opencv with cmake 
```
# ~/opencv
mkdir build
```

```
.
└── opencv
    ├── build
    └── source
        ├── opencv
        └── opencv_contrib
```


```bash
cd build 
cmake -DCMAKE_BUILD_TYPE=Release \
-DBUILD_EXAMPLES=ON \
-DOPENCV_EXTRA_MODULES_PATH="../source/opencv_contrib/modules/" \
-DCMAKE_INSTALL_PREFIX="../opencv/opencv_libs" ../source/opencv 


cmake 
-G "Unix Makefiles" ..
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D BUILD_EXAMPLES=ON \
-D WITH_CUDA=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \


cmake -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_TYPE=Release   
  -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules  
  -D WITH_CUDA=ON 
  -D WITH_TBB=ON 
  -D ENABLE_FAST_MATH=1 
  -D WITH_OPENMP=ON 
  -D WITH_CUFFT=ON 
  -D WITH_CUBLAS=ON 
  -D CUDA_FAST_MATH=1 
  -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..



//cmake输出中包含 以下内容则可以使用CUDA
NVIDIA CUDA:                   YES (ver 11.4, CUFFT CUBLAS FAST_MATH)
--     NVIDIA GPU arch:             35 37 50 52 60 61 70 75 80 86
--     NVIDIA PTX archs:
-- 
--   cuDNN:                         YES (ver 8.2.1)

```


Build Directory:
- .. - Parent directory containing the source code (CMakeLists.txt)
- -G "Unix Makefiles"   - generator for this project 


Build & installation configuration
- -D CMAKE_BUILD_TYPE=RELEASE - Builds an optimized release version (vs debug)
- -D CMAKE_INSTALL_PREFIX=/usr/local - Installation directory (standard Linux location)
- -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules - Path to OpenCV's extra/contrib modules (non-free/experimental features)

- -D BUILD_opencv_world=OFF   - It combines all OpenCV libraries (core, imgproc, dnn, cudaarithm, etc.) into a single dynamic/static library (e.g., libopencv_world.so). The advantage is easier deployment; the disadvantages are longer compile times and the need to recompile the entire library for any minor change.

Generate Example 
- -D BUILD_EXAMPLES=ON - Compiles OpenCV example programs
- -D INSTALL_C_EXAMPLES=ON
    - Copies C and C++ example source code and binaries to the installation directory. Valuable for learning and quick testing of OpenCV features.
- -D INSTALL_PYTHON_EXAMPLES=ON
 
 Developer convenience
- -D OPENCV_GENERATE_PKGCONFIG=ON 
    - Essential for Linux developers. Allows easy compilation of OpenCV programs with pkg-config --cflags --libs opencv4. Makes integration with build systems like Makefiles and autotools much simpler.
    - Without this: You need to manually specify include paths (-I/usr/local/include/opencv4) and library paths (-L/usr/local/lib -lopencv_core -lopencv_imgproc ...).
    - With this: Simple compilation becomes: `g++ myprogram.cpp -o myprogram `pkg-config --cflags --libs opencv4``
- -D ENABLE_CXX11=1
    - Ensures modern C++11 features are enabled. Important for compatibility with newer compilers and certain C++11-dependent libraries. Usually auto-detected, but explicit setting prevents issues.

Algorithm availability
- -D OPENCV_ENABLE_NONFREE=True
    - Required for using SIFT, SURF, and other patented algorithms. Without this, these features are disabled due to licensing restrictions. Set to True if you need these algorithms for research/internal use (understand licensing implications).

Python configuration
	-D HAVE_opencv_python3=ON \
	-D PYTHON_EXECUTABLE=~/.virtualenvs/opencv_cuda/bin/python \
---


CUDA Configuration:

Essential Options: 
- -D WITH_CUDA=ON 
    - Enables CUDA GPU acceleration
- -D WITH_CUDNN=ON \
    - enable cuDNN (CUDA Deep Neural Network library) support in OpenCV. 
- -D OPENCV_DNN_CUDA=ON 
    - Strongly recommended to enable. This enables the CUDA backend for OpenCV's deep neural network module (dnn). When enabled, neural network models (like YOLO) loaded with the dnn::Net class will leverage the GPU for inference, resulting in significant speedups.


Performance Optimization: 
- -D CUDA_ARCH_BIN=your_compute_capability.   
    - This option specifies which GPU architectures to generate Cubin binary code for. The best practice is to specify only the compute capability of your current GPU, for example, 7.5 for your T1200. This maximizes performance for your specific card while avoiding excessively long compile times and oversized binaries.
- -D ENABLE_FAST_MATH=1 
    - Enables fast math operations (speed > precision)
- -D CUDA_FAST_MATH=1
    - Enables CUDA-specific fast math optimizations
- -D WITH_CUBLAS=1 
    - Enables cuBLAS library for BLAS operations on GPU

Optional: 
- - -D BUILD_CUDA_STUBS=ON
    - Primarily used for legacy OpenCV versions or special cross-compilation scenarios (e.g., packaging on a Linux system without a GPU). For standard desktop environments compiling CUDA-accelerated OpenCV, it should usually be set to OFF or omitted (default is OFF).



- CUDA_ARCH_BIN 的选项在里面填入你当前电脑显卡的计算系数。
    - 可以在这里找到对应显卡的算力：Nvidia显卡算力表  https://developer.nvidia.com/cuda/gpus\
    - 访问 NVIDIA-Your GPU Compute Capability，下滑找到CUDA-Enabled GeForce and TiTAN Products后点击并查看自己显卡算力，下一步需要填写CUDA_ARCH_BIN参数。
    - 使用nvidia-smi -L 查看NVIDIA-GPU型号。到https://developer.nvidia.com/cuda-gpus 中查找GPU型号对应的compute capability，即CUDA_ARCH_BIN。 比如我的电脑是T1200，CUDA_ARCH_BIN=7.5

![](image/Pasted%20image%2020260211144404.png)

![](image/Pasted%20image%2020260211153820.png)

----

### 1.2.2 after cmake, make it again
```bash
cd build
make -j$(sysctl -n hw.ncpu) 
# make -j16 indicate excute 16 jobs simultanously 

#  then
make install 


# then 
//为opencv库创建所有必要的链接和缓存
sudo ldconfig 

```

all complied files is saved in  build/install


### 1.2.3 Opencv environment variable 

OPENCV_DIR:C:\opencv\build\x64\vc14；

```bash
# Step 1: Configure the library path
# What it does: Creates/edits a configuration file for the dynamic linker/loader
# 执行此命令后打开的可能是一个空白的文件，不用管，只需要在文件末尾添加:
sudo gedit /etc/ld.so.conf.d/opencv.conf

# File content to add:
/usr/local/lib
# This tells the system to look for shared libraries (.so files) in /usr/local/lib where OpenCV libraries were installed.


# Updates the shared library cache
# Rebuilds /etc/ld.so.cache
# Makes the system immediately aware of new libraries in /usr/local/lib
# Without this, you'd get error while loading shared libraries when running OpenCV programs
sudo ldconfig 

# Check if OpenCV libraries are found by the linker
ldconfig -p | grep opencv



# Step 2: Configure pkg-config path
sudo gedit /etc/bash.bashrc

# File content to add:
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig  
export PKG_CONFIG_PATH

# update 
source /etc/bash.bashrc

# Check if pkg-config can find OpenCV
pkg-config --modversion opencv4

```
## 1.3 Test 

### 1.3.1 

在opencv/samples/gpu目录下，执行任何一个.exe程序
```
pkg-config opencv --modversion
$4.5.5
```

/usr/local/lib/pkgconfig/ 文件夹下包含opencv.pc文件
```
prefix=/usr/local
exec_prefix=${prefix}
includedir=/usr/local/include
libdir=/usr/local/lib
 
Name: OpenCV
Description: Open Source Computer Vision Library
Version: 4.5.5
Libs: -L${exec_prefix}/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dpm -lopencv_face -lopencv_photo -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_mcc -lopencv_sfm -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lopencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core -lopencv_viz -lopencv_gapi -lopencv_cudev -lopencv_rapid -lopencv_stereo -lopencv_barcode -lopencv_quality -lopencv_alphamat -lopencv_cudacodec -lopencv_cudaarithm -lopencv_cudabgsegm -lopencv_cudalegacy -lopencv_cudastereo -lopencv_cudafilters -lopencv_cudaimgproc -lopencv_cudaoptflow -lopencv_cudawraping -lopencv_dnn_supress -lopencv_cudaobjdetect -lopencv_wechat_qrcode -lopencv_cudafeatures2d -lopencv_intensity_transform 
Libs.private: -ldl -lm -lpthread -lrt
Cflags: -I${includedir}
```


### 1.3.2 使用OpenCV处理图像


1. 创建新文件夹，我命名为OpenCV-Test
2. 进入文件夹后右键打开终端
3. touch main.cpp 创建名为main.cpp的文件
4. 右键main.cpp，使用IDE打开
5. 复制代码：

```
#include <opencv2/opencv.hpp> 
#include <iostream> 

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
 //读取照片
 Mat image = imread("OpenCV_Logo.png");

 //检测失误
 if (image.empty()) 
 {
  cout << "Could not open or find the image" << endl;
  cin.get(); //等待键盘输入
  return -1;
 }

 String windowName = "OpenCV Test";	   //窗口名称
 namedWindow(windowName); 		   //创建新窗口
 imshow(windowName, image);		   //使用新窗口显示照片
 waitKey(0); 				   //等待键盘输入
 destroyWindow(windowName);		   //关闭所有窗口
 return 0;
}
```


6. 网上随意下载张图片，放入与main.cpp相同的文件夹中。
7. 更改图片路径。
8. g++编译

```
g++ -o main main.cpp `pkg-config --libs --cflags opencv`

g++ -std=c++11 *.cpp 'pkg-config --libs --cflags opencv' -o out
```


9. 运行程序
./main
10. 没有报错就是安装成功了

### 1.3.3 验证 C++ 是否支持 CUDA 的 OpeCV

```
创建 test_opencv.cpp
sudo nano test_opencv.cpp
```

test_opencv.cpp
```
#include <opencv2/cvconfig.h>
#include <opencv2/opencv.hpp>
#include <iostream>
int main() {
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
#ifdef HAVE_CUDA
    std::cout << "CUDA is available." << std::endl;
#else
    std::cout << "CUDA is NOT available." << std::endl;
#endif
    return 0;
}
————————————————
版权声明：本文为CSDN博主「Natsuagin」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Natsuago/article/details/145785243
```

执行以下命令编译test_opencv.cpp：

```
g++ test_opencv.cpp -o test_opencv `pkg-config --cflags --libs opencv4` -I/usr/local/include/opencv4/opencv2 -L/usr/local/lib -Wl,-rpath,/usr/local/lib

————————————————
版权声明：本文为CSDN博主「Natsuagin」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Natsuago/article/details/145785243
```


运行编译结果：
```
./test_opencv

```

如果显示opencv版本和CUDA is available.说明 OpenCV 已成功启用 CUDA（例如下图）。

![](image/Pasted%20image%2020260211154152.png)

