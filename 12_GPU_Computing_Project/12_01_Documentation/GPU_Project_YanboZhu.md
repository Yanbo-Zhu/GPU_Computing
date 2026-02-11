

# 1 Setup


## 1.1 install dependency


```
sudo apt update
sudo apt install -y build-essential cmake git unzip pkg-config 
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev ffmpeg
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install -y libxvidcore-dev libx264-dev libx265-dev
sudo apt install -y libgtk-3-dev libcanberra-gtk3-dev
sudo apt install -y libatlas-base-dev gfortran
sudo apt install -y python3-dev python3-pip
pip3 install numpy



# 下面的不用 
sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libatlas-base-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    liblapacke-dev \
    gfortran \
    python3-dev \
    python3-numpy \
    ffmpeg
```


![](image/Pasted%20image%2020260211215447.png)


## 1.2 Install Opencv

```bash
mkdir -r opencv/source

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

cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=../source/opencv_contrib/modules \
-D BUILD_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-G "Unix Makefiles" \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D CUDA_ARCH_BIN=6.1 \
-D HAVE_OPENCV_PYTHON3=ON \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
-D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
../source/opencv



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
- -D HAVE_opencv_python3=ON \
- -D PYTHON_EXECUTABLE=~/.virtualenvs/opencv_cuda/bin/python \
- -D PYTHON3_EXECUTABLE=$(which python3) \
- -D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
- -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())")


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
        - GPU 0: NVIDIA TITAN Xp (UUID: GPU-4c49dbc7-5541-25eb-3ab1-e963ff6602d5)  对应 6.1

![](image/Pasted%20image%2020260211144404.png)

![](image/Pasted%20image%2020260211153820.png)

----
after run cmake command , we get

1 in the macOS 

```bash
-- SYCL/OpenCL samples are skipped: SYCL SDK is required
--    - check configuration of SYCL_DIR/SYCL_ROOT/CMAKE_MODULE_PATH
--    - ensure that right compiler is selected from SYCL SDK (e.g, clang++): CMAKE_CXX_COMPILER=/usr/bin/c++
CMake Warning (dev) at samples/CMakeLists.txt:13 (install):
  Policy CMP0177 is not set: install() DESTINATION paths are normalized.  Run
  "cmake --help-policy CMP0177" for policy details.  Use the cmake_policy
  command to set the policy and suppress this warning.
Call Stack (most recent call first):
  samples/CMakeLists.txt:54 (ocv_install_example_src)
This warning is for project developers.  Use -Wno-dev to suppress it.

--
-- General configuration for OpenCV 4.13.0 =====================================
--   Version control:               4.13.0
--
--   Extra modules:
--     Location (extra):            /Users/yanbo/Documents/Code_Storage/opencv/source/opencv_contrib/modules
--     Version control (extra):     4.13.0
--
--   Platform:
--     Timestamp:                   2026-02-11T19:21:04Z
--     Host:                        Darwin 22.6.0 x86_64
--     CMake:                       4.2.3
--     CMake generator:             Unix Makefiles
--     CMake build tool:            /usr/bin/make
--     Configuration:               RELEASE
--     Algorithm Hint:              ALGO_HINT_ACCURATE
--
--   CPU/HW features:
--     Baseline:                    SSE SSE2 SSE3 SSSE3 SSE4_1
--       requested:                 DETECT
--     Dispatched code generation:  SSE4_2 AVX FP16 AVX2 AVX512_SKX
--       requested:                 SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX
--       SSE4_2 (2 files):          + POPCNT SSE4_2
--       AVX (10 files):            + POPCNT SSE4_2 AVX
--       FP16 (1 files):            + POPCNT SSE4_2 AVX FP16
--       AVX2 (39 files):           + POPCNT SSE4_2 AVX FP16 AVX2 FMA3
--       AVX512_SKX (10 files):     + POPCNT SSE4_2 AVX FP16 AVX2 FMA3 AVX_512F AVX512_COMMON AVX512_SKX
--
--   C/C++:
--     Built as dynamic libs?:      YES
--     C++ standard:                11
--     C++ Compiler:                /usr/bin/c++  (ver 14.0.0.14000029)
--     C++ flags (Release):         -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments -Wno-semicolon-before-method-body -ffunction-sections -fdata-sections  -fvisibility=hidden -fvisibility-inlines-hidden -Wno-deprecated-copy -O3 -DNDEBUG  -DNDEBUG
--     C++ flags (Debug):           -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments -Wno-semicolon-before-method-body -ffunction-sections -fdata-sections  -fvisibility=hidden -fvisibility-inlines-hidden -Wno-deprecated-copy -g  -O0 -DDEBUG -D_DEBUG
--     C Compiler:                  /usr/bin/cc
--     C flags (Release):           -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments -Wno-semicolon-before-method-body -ffunction-sections -fdata-sections  -fvisibility=hidden -fvisibility-inlines-hidden -O3 -DNDEBUG  -DNDEBUG
--     C flags (Debug):             -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments -Wno-semicolon-before-method-body -ffunction-sections -fdata-sections  -fvisibility=hidden -fvisibility-inlines-hidden -g  -O0 -DDEBUG -D_DEBUG
--     Linker flags (Release):      -L/usr/local/opt/ruby/lib  -Wl,-dead_strip
--     Linker flags (Debug):        -L/usr/local/opt/ruby/lib  -Wl,-dead_strip
--     ccache:                      NO
--     Precompiled headers:         NO
--     Extra dependencies:
--     3rdparty dependencies:
--
--   OpenCV modules:
--     To be built:                 alphamat aruco bgsegm bioinspired calib3d ccalib core datasets dnn dnn_objdetect dnn_superres dpm face features2d flann freetype fuzzy gapi hfs highgui img_hash imgcodecs imgproc intensity_transform java line_descriptor mcc ml objdetect optflow phase_unwrapping photo plot quality rapid reg rgbd saliency shape signal stereo stitching structured_light superres surface_matching text tracking ts video videoio videostab wechat_qrcode xfeatures2d ximgproc xobjdetect xphoto
--     Disabled:                    world
--     Disabled by dependency:      -
--     Unavailable:                 cannops cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev cvv fastcv hdf julia matlab ovis python2 python3 sfm viz
--     Applications:                tests perf_tests examples apps
--     Documentation:               NO
--     Non-free algorithms:         YES
--
--   GUI:                           COCOA
--     Cocoa:                       YES
--     VTK support:                 NO
--
--   Media I/O:
--     ZLib:                        build (ver 1.3.1)
--     JPEG:                        build-libjpeg-turbo (ver 3.1.2-70)
--       SIMD Support Request:      YES
--       SIMD Support:              YES
--     WEBP:                        build (ver decoder: 0x0210, encoder: 0x0210, demux: 0x0107)
--     AVIF:                        avif (ver 0.11.1)
--     PNG:                         build (ver 1.6.53)
--       SIMD Support Request:      YES
--       SIMD Support:              YES (Intel SSE)
--     TIFF:                        build (ver 42 - 4.7.1)
--     JPEG 2000:                   build (ver 2.5.3)
--     OpenEXR:                     OpenEXR::OpenEXR (ver 3.4.4)
--     GIF:                         YES
--     HDR:                         YES
--     SUNRASTER:                   YES
--     PXM:                         YES
--     PFM:                         YES
--
--   Video I/O:
--     FFMPEG:                      YES
--       avcodec:                   YES (62.11.100)
--       avformat:                  YES (62.3.100)
--       avutil:                    YES (60.8.100)
--       swscale:                   YES (9.1.100)
--       avdevice:                  YES (62.1.100)
--     GStreamer:                   NO
--     AVFoundation:                YES
--     Orbbec:                      NO
--
--   Parallel framework:            GCD
--
--   Trace:                         YES (with Intel ITT(3.25.4))
--
--   Other third-party libraries:
--     Intel IPP:                   2021.9.1 [2021.9.1]
--            at:                   /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/build/3rdparty/ippicv/ippicv_mac/icv
--     Intel IPP IW:                sources (2021.9.1)
--               at:                /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/build/3rdparty/ippicv/ippicv_mac/iw
--     Lapack:                      YES (-framework Accelerate)
--     Eigen:                       YES (ver 5.0.1)
--     Custom HAL:                  YES (ipp (ver 0.0.1))
--     Protobuf:                    build (3.19.1)
--     Flatbuffers:                 builtin/3rdparty (25.9.23)
--
--   OpenCL:                        YES (no extra features)
--     Include path:                NO
--     Link libraries:              -framework OpenCL
--
--   Python (for build):            /usr/local/bin/python3
--
--   Java:
--     ant:                         NO
--     Java:                        YES (ver 11.0.17)
--     JNI:                         /Library/Java/JavaVirtualMachines/jdk-11.0.17.jdk/Contents/Home/include /Library/Java/JavaVirtualMachines/jdk-11.0.17.jdk/Contents/Home/include/darwin /Library/Java/JavaVirtualMachines/jdk-11.0.17.jdk/Contents/Home/include
--     Java wrappers:               YES (JAVA)
--     Java tests:                  NO
--
--   Install to:                    /usr/local
-- -----------------------------------------------------------------
--
-- Configuring done (172.7s)
-- Generating done (10.1s)
-- Build files have been written to: /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/build
 yanbo@Yanbos-MacBook-Pro  ~/Documents/Code_Storage/opencv/source/opencv/build  ➦ fe38fc608f  cmake .. \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D BUILD_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_ENABLE_NONFREE=ON
CMake Deprecation Warning at CMakeLists.txt:25 (cmake_minimum_required):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- Detected processor: x86_64
-- Looking for ccache - not found
-- AVX512_KNM is not supported by C++ compiler
-- libjpeg-turbo: VERSION = 3.1.2, BUILD = opencv-4.13.0-libjpeg-turbo
-- Performing Test HAVE_BUILTIN_CTZL
-- Performing Test HAVE_BUILTIN_CTZL - Success
-- CMAKE_ASM_NASM_COMPILER = /usr/local/bin/nasm
-- CMAKE_ASM_NASM_OBJECT_FORMAT = macho64
-- CMAKE_ASM_NASM_FLAGS =  -DMACHO -D__x86_64__ -DPIC
-- SIMD extensions: x86_64 (WITH_SIMD = 1)
-- Could NOT find OpenJPEG (minimal suitable version: 2.0, recommended version >= 2.3.1). OpenJPEG will be built from sources
-- OpenJPEG: VERSION = 2.5.3, BUILD = opencv-4.13.0-openjp2-2.5.3
-- OpenJPEG libraries will be built from sources: libopenjp2 (version "2.5.3")
-- found Intel IPP (ICV version): 2021.9.1 [2021.9.1]
-- at: /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/build/3rdparty/ippicv/ippicv_mac/icv
-- found Intel IPP Integration Wrappers sources: 2021.9.1
-- at: /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/build/3rdparty/ippicv/ippicv_mac/iw
-- LAPACK(Unknown): LAPACK_LIBRARIES: -framework Accelerate
-- Looking for Accelerate/Accelerate.h
-- Looking for Accelerate/Accelerate.h - found
-- Looking for Accelerate/Accelerate.h
-- Looking for Accelerate/Accelerate.h - found
-- LAPACK(Unknown): Support is enabled.
-- Could NOT find Pylint (missing: PYLINT_EXECUTABLE)
-- Could NOT find Flake8 (missing: FLAKE8_EXECUTABLE)
-- VTK is not found. Please set -DVTK_DIR in CMake to VTK build directory, or to VTK install subdirectory with VTKConfig.cmake file
-- Checking for module 'gstreamer-base-1.0'
--   Package 'gstreamer-base-1.0' not found
-- Checking for module 'gstreamer-app-1.0'
--   Package 'gstreamer-app-1.0' not found
-- Checking for module 'gstreamer-riff-1.0'
--   Package 'gstreamer-riff-1.0' not found
-- Checking for module 'gstreamer-pbutils-1.0'
--   Package 'gstreamer-pbutils-1.0' not found
-- Checking for module 'gstreamer-video-1.0'
--   Package 'gstreamer-video-1.0' not found
-- Checking for module 'gstreamer-audio-1.0'
--   Package 'gstreamer-audio-1.0' not found
-- freetype2:   YES (ver 24.3.18)
-- harfbuzz:    YES (ver 5.3.1)
-- Could NOT find HDF5 (missing: HDF5_LIBRARIES HDF5_INCLUDE_DIRS)
-- Julia not found. Not compiling Julia Bindings.
-- Module opencv_ovis disabled because OGRE3D was not found
-- No preference for use of exported gflags CMake configuration set, and no hints for include/library directories provided. Defaulting to preferring an installed/exported gflags CMake configuration if available.
-- Failed to find installed gflags CMake configuration, searching for gflags build directories exported with CMake.
-- Failed to find gflags - Failed to find an installed/exported CMake configuration for gflags, will perform search for installed gflags components.
-- Failed to find gflags - Could not find gflags include directory, set GFLAGS_INCLUDE_DIR to directory containing gflags/gflags.h
-- Failed to find glog - Could not find glog include directory, set GLOG_INCLUDE_DIR to directory containing glog/logging.h
-- Module opencv_sfm disabled because the following dependencies are not found: Glog/Gflags
-- Tesseract:   YES (ver 5.2.0)
-- Allocator metrics storage type: 'long long'
-- Excluding from source files list: <BUILD>/modules/core/test/test_intrin256.lasx.cpp
-- Excluding from source files list: modules/imgproc/src/imgwarp.lasx.cpp
-- Excluding from source files list: modules/imgproc/src/resize.lasx.cpp
-- Registering hook 'INIT_MODULE_SOURCES_opencv_dnn': /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/modules/dnn/cmake/hooks/INIT_MODULE_SOURCES_opencv_dnn.cmake
-- opencv_dnn: filter out ocl4dnn source code
-- opencv_dnn: filter out cuda4dnn source code
-- Excluding from source files list: <BUILD>/modules/dnn/layers/layers_common.rvv.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/layers_common.lasx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/layers_common.neon.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/layers_common.sve.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/int8layers/layers_common.rvv.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/int8layers/layers_common.lasx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/int8layers/layers_common.neon.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/conv_block.neon.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/conv_block.neon_fp16.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/conv_depthwise.rvv.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/conv_depthwise.lasx.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/fast_gemm_kernels.neon.cpp
-- Excluding from source files list: <BUILD>/modules/dnn/layers/cpu_kernels/fast_gemm_kernels.lasx.cpp
-- highgui: using builtin backend: COCOA
-- Use autogenerated whitelist /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/build/modules/js_bindings_generator/whitelist.json
-- Set Cleaners to True
-- Found 'misc' Python modules from /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/modules/python/package/extra_modules
-- Found 'mat_wrapper;utils' Python modules from /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/modules/core/misc/python/package
-- Found 'gapi' Python modules from /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/modules/gapi/misc/python/package
CMake Deprecation Warning at samples/sycl/CMakeLists.txt:24 (cmake_policy):
  Compatibility with CMake < 3.10 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value.  Or, use the <min>...<max> syntax
  to tell CMake that the project requires at least <min> but has been updated
  to work with policies introduced by <max> or earlier.


-- SYCL/OpenCL samples are skipped: SYCL SDK is required
--    - check configuration of SYCL_DIR/SYCL_ROOT/CMAKE_MODULE_PATH
--    - ensure that right compiler is selected from SYCL SDK (e.g, clang++): CMAKE_CXX_COMPILER=/usr/bin/c++
CMake Warning (dev) at samples/CMakeLists.txt:13 (install):
  Policy CMP0177 is not set: install() DESTINATION paths are normalized.  Run
  "cmake --help-policy CMP0177" for policy details.  Use the cmake_policy
  command to set the policy and suppress this warning.
Call Stack (most recent call first):
  samples/CMakeLists.txt:54 (ocv_install_example_src)
This warning is for project developers.  Use -Wno-dev to suppress it.

--
-- General configuration for OpenCV 4.13.0 =====================================
--   Version control:               4.13.0
--
--   Extra modules:
--     Location (extra):            /Users/yanbo/Documents/Code_Storage/opencv/source/opencv_contrib/modules
--     Version control (extra):     4.13.0
--
--   Platform:
--     Timestamp:                   2026-02-11T19:21:04Z
--     Host:                        Darwin 22.6.0 x86_64
--     CMake:                       4.2.3
--     CMake generator:             Unix Makefiles
--     CMake build tool:            /usr/bin/make
--     Configuration:               RELEASE
--     Algorithm Hint:              ALGO_HINT_ACCURATE
--
--   CPU/HW features:
--     Baseline:                    SSE SSE2 SSE3 SSSE3 SSE4_1
--       requested:                 DETECT
--     Dispatched code generation:  SSE4_2 AVX FP16 AVX2 AVX512_SKX
--       requested:                 SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX
--       SSE4_2 (2 files):          + POPCNT SSE4_2
--       AVX (10 files):            + POPCNT SSE4_2 AVX
--       FP16 (1 files):            + POPCNT SSE4_2 AVX FP16
--       AVX2 (39 files):           + POPCNT SSE4_2 AVX FP16 AVX2 FMA3
--       AVX512_SKX (10 files):     + POPCNT SSE4_2 AVX FP16 AVX2 FMA3 AVX_512F AVX512_COMMON AVX512_SKX
--
--   C/C++:
--     Built as dynamic libs?:      YES
--     C++ standard:                11
--     C++ Compiler:                /usr/bin/c++  (ver 14.0.0.14000029)
--     C++ flags (Release):         -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments -Wno-semicolon-before-method-body -ffunction-sections -fdata-sections  -fvisibility=hidden -fvisibility-inlines-hidden -Wno-deprecated-copy -O3 -DNDEBUG  -DNDEBUG
--     C++ flags (Debug):           -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments -Wno-semicolon-before-method-body -ffunction-sections -fdata-sections  -fvisibility=hidden -fvisibility-inlines-hidden -Wno-deprecated-copy -g  -O0 -DDEBUG -D_DEBUG
--     C Compiler:                  /usr/bin/cc
--     C flags (Release):           -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments -Wno-semicolon-before-method-body -ffunction-sections -fdata-sections  -fvisibility=hidden -fvisibility-inlines-hidden -O3 -DNDEBUG  -DNDEBUG
--     C flags (Debug):             -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments -Wno-semicolon-before-method-body -ffunction-sections -fdata-sections  -fvisibility=hidden -fvisibility-inlines-hidden -g  -O0 -DDEBUG -D_DEBUG
--     Linker flags (Release):      -L/usr/local/opt/ruby/lib  -Wl,-dead_strip
--     Linker flags (Debug):        -L/usr/local/opt/ruby/lib  -Wl,-dead_strip
--     ccache:                      NO
--     Precompiled headers:         NO
--     Extra dependencies:
--     3rdparty dependencies:
--
--   OpenCV modules:
--     To be built:                 alphamat aruco bgsegm bioinspired calib3d ccalib core datasets dnn dnn_objdetect dnn_superres dpm face features2d flann freetype fuzzy gapi hfs highgui img_hash imgcodecs imgproc intensity_transform java line_descriptor mcc ml objdetect optflow phase_unwrapping photo plot quality rapid reg rgbd saliency shape signal stereo stitching structured_light superres surface_matching text tracking ts video videoio videostab wechat_qrcode xfeatures2d ximgproc xobjdetect xphoto
--     Disabled:                    world
--     Disabled by dependency:      -
--     Unavailable:                 cannops cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev cvv fastcv hdf julia matlab ovis python2 python3 sfm viz
--     Applications:                tests perf_tests examples apps
--     Documentation:               NO
--     Non-free algorithms:         YES
--
--   GUI:                           COCOA
--     Cocoa:                       YES
--     VTK support:                 NO
--
--   Media I/O:
--     ZLib:                        build (ver 1.3.1)
--     JPEG:                        build-libjpeg-turbo (ver 3.1.2-70)
--       SIMD Support Request:      YES
--       SIMD Support:              YES
--     WEBP:                        build (ver decoder: 0x0210, encoder: 0x0210, demux: 0x0107)
--     AVIF:                        avif (ver 0.11.1)
--     PNG:                         build (ver 1.6.53)
--       SIMD Support Request:      YES
--       SIMD Support:              YES (Intel SSE)
--     TIFF:                        build (ver 42 - 4.7.1)
--     JPEG 2000:                   build (ver 2.5.3)
--     OpenEXR:                     OpenEXR::OpenEXR (ver 3.4.4)
--     GIF:                         YES
--     HDR:                         YES
--     SUNRASTER:                   YES
--     PXM:                         YES
--     PFM:                         YES
--
--   Video I/O:
--     FFMPEG:                      YES
--       avcodec:                   YES (62.11.100)
--       avformat:                  YES (62.3.100)
--       avutil:                    YES (60.8.100)
--       swscale:                   YES (9.1.100)
--       avdevice:                  YES (62.1.100)
--     GStreamer:                   NO
--     AVFoundation:                YES
--     Orbbec:                      NO
--
--   Parallel framework:            GCD
--
--   Trace:                         YES (with Intel ITT(3.25.4))
--
--   Other third-party libraries:
--     Intel IPP:                   2021.9.1 [2021.9.1]
--            at:                   /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/build/3rdparty/ippicv/ippicv_mac/icv
--     Intel IPP IW:                sources (2021.9.1)
--               at:                /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/build/3rdparty/ippicv/ippicv_mac/iw
--     Lapack:                      YES (-framework Accelerate)
--     Eigen:                       YES (ver 5.0.1)
--     Custom HAL:                  YES (ipp (ver 0.0.1))
--     Protobuf:                    build (3.19.1)
--     Flatbuffers:                 builtin/3rdparty (25.9.23)
--
--   OpenCL:                        YES (no extra features)
--     Include path:                NO
--     Link libraries:              -framework OpenCL
--
--   Python (for build):            /usr/local/bin/python3
--
--   Java:
--     ant:                         NO
--     Java:                        YES (ver 11.0.17)
--     JNI:                         /Library/Java/JavaVirtualMachines/jdk-11.0.17.jdk/Contents/Home/include /Library/Java/JavaVirtualMachines/jdk-11.0.17.jdk/Contents/Home/include/darwin /Library/Java/JavaVirtualMachines/jdk-11.0.17.jdk/Contents/Home/include
--     Java wrappers:               YES (JAVA)
--     Java tests:                  NO
--
--   Install to:                    /usr/local
-- -----------------------------------------------------------------
--
-- Configuring done (16.0s)
-- Generating done (8.6s)
-- Build files have been written to: /Users/yanbo/Documents/Code_Storage/opencv/source/opencv/build
```


---

2 in the linux 

```
CMake Warning at samples/samples_utils.cmake:10 (add_executable):
  Cannot generate a safe runtime search path for target
  example_opencl_opencl-opencv-interop because files in some directories may
  conflict with libraries in implicit directories:

    runtime library [libOpenCL.so.1] in /usr/lib/x86_64-linux-gnu may be hidden by files in:
      /usr/local/cuda/lib64

  Some of these libraries may not be found correctly.
Call Stack (most recent call first):
  samples/opencl/CMakeLists.txt:33 (ocv_define_sample)
```



### 1.2.2 after cmake, make it again
```bash
cd build

# make -j16 indicate excute 16 jobs simultanously 
# Compilation only - files are in build/ directory
#  in maxos
make -j$(sysctl -n hw.ncpu) 

# in linux 
make -j$(nproc)

#  MUST RUN THIS - copies files to /usr/local/
make install 



```

all complied files is saved in  build/install

---
after runs cmake 
```
build/
├── lib/                    # 编译生成的临时库文件 (.dylib)
├── bin/                    # 编译生成的临时可执行文件
├── modules/                # 每个模块的编译中间文件
│   ├── core/              # core 模块的 .o 文件
│   ├── imgproc/           # imgproc 模块的 .o 文件
│   └── ...
├── libopencv_core.4.13.0.dylib  # 编译生成的库文件
├── opencv_version         # 版本测试程序
└── CMakeCache.txt         # CMake 缓存配置
```


After running sudo make install, files are copied HERE /usr/local/ (set by -D CMAKE_INSTALL_PREFIX=/usr/local)

```
/usr/local/
├── lib/                    # ✅ FINAL LOCATION - Libraries
│   ├── libopencv_core.4.13.0.dylib
│   ├── libopencv_imgproc.4.13.0.dylib
│   ├── libopencv_dnn.4.13.0.dylib
│   ├── libopencv_contrib*.dylib  # Contrib modules
│   └── pkgconfig/         # pkg-config configuration
│       └── opencv4.pc
│
├── include/               # ✅ FINAL LOCATION - Headers
│   └── opencv4/
│       └── opencv2/
│           ├── opencv.hpp
│           ├── core.hpp
│           └── ...
│
├── bin/                   # Executables
│   ├── opencv_version
│   └── opencv_*_demo
│
└── share/                # Examples and documentation
    └── opencv4/
        └── examples/
```


### 1.2.3 Opencv environment variable in Linux

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



##### Step3:  Test OpenCV in Python
python3 -c "import cv2; print(cv2.__version__)"

```


### 1.2.4 Opencv environment variable in MacOS

Step 1: 配置动态库路径

```
# 1. macOS 不需要单独配置库路径，/usr/local/lib 默认已在搜索路径中
# 验证方式：
otool -L /usr/local/lib/libopencv_core.dylib   # 检查库依赖

It returns
/usr/local/lib/libopencv_core.dylib:
	@rpath/libopencv_core.413.dylib (compatibility version 413.0.0, current version 4.13.0)
	/System/Library/Frameworks/OpenCL.framework/Versions/A/OpenCL (compatibility version 1.0.0, current version 1.0.0)
	/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate (compatibility version 1.0.0, current version 4.0.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1319.0.0)
	/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1300.32.0)

```

如果确实需要添加自定义库路径：
```
# 编辑动态链接器环境变量（临时生效）
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH

# 永久生效（添加到 ~/.zshrc）
echo 'export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```

---

Step 2: 配置 pkg-config 路径
```
# 1. 编辑 zsh 配置文件
nano ~/.zshrc
# 或使用其他编辑器：vim ~/.zshrc 或 code ~/.zshrc

# 2. 添加以下内容：
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"

# 3. 使配置生效
source ~/.zshrc

# 4. 验证
echo $PKG_CONFIG_PATH    # 应包含 /usr/local/lib/pkgconfig
```


---

Step3: 验证 OpenCV 安装
```
# 1. 检查 pkg-config 是否能找到 OpenCV
pkg-config --modversion opencv4

# 如果失败，尝试 opencv（无数字）：
pkg-config --modversion opencv

# 2. 检查库文件是否存在
ls -la /usr/local/lib/libopencv_*    # 查看安装的 OpenCV 库文件

# 3. 检查 Python 绑定（如果编译了 Python 支持）
python3 -c "import cv2; print(cv2.__version__)"

# 4. 查看库依赖
otool -L /usr/local/lib/libopencv_core.dylib | grep cuda  # 检查 CUDA 支持（如果启用）
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

```c++
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
g++ main.cpp `pkg-config --libs --cflags opencv4` -o main

g++ main.cpp 'pkg-config --libs --cflags opencv4'  -o out -std=c++11 
```


9. 运行程序
./main
10. 没有报错就是安装成功了

### 1.3.3 验证 C++ 是否支持 CUDA 的 OpenCV

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

