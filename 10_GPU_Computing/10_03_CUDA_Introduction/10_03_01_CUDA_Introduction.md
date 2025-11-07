
• Programming model of CUDA
• Mapping of programming model to the hardware architectures

# 1 Host Vs. Device


• The CUDA Programming model distinguishes between host and device code
• The physically separate device operates as a parallel coprocessor to the host running the main C++ program
• The host manages the execution of code on the device

Memory
• Host (CPU) and Device (GPU) have physically separate memory
• The host allocates and deallocates memory on the device
• Data is transferred explicitly between host and device memory (we will late in the class learn about the automatically managed “unified memory”)


![[Pasted image 20251107145919.png]]

# 2 CUDA C Program Structure

• Most simple CUDA programs follows a similar structure:
	• Data is allocated and copied to the device
	• Computation on the data is performed in the “kernel” function
	• Data is copied back to the host and deallocated


```
int main() {
// 1. allocate memory on device
// 2. copy data from host to device
// 3. launches multiple instances of
// execution “kernel” on device
// 4. copy data from device back to host
// 5. de-allocate memory on device
}
```

## 2.1 Device Memory Allocation & Deallocation

![[Pasted image 20251107150235.png]]

• We allocate memory on the device using cudaMalloc:
• The function takes as arguments:
1. the address of a pointer;
2. the size in bytes.
• We deallocate device memory using cudaFree:
• Many CUDA functions return an error code, which should be cudaSuccess


## 2.2 Copy Data Between Host and Device

• To copy data between host and device we use cudaMemcpy:
![[Pasted image 20251107150323.png]]


• The function takes as arguments:
1. the destination pointer;
2. the source pointer;
3. the size in bytes;
4. the kind of copy (host-host, host-device, device-host, device-device).


## 2.3 Launching Kernel Functions

• The kernel is a function that is executed in parallel on the device
• It is written with special rules, so that it can be executed by many threads
• It is also launched specially by specifying the launch configuration, the number of threads executing the kernel and how they are organized
• Launching a kernel is done with a special syntax:

![[Pasted image 20251107152615.png]]


## 2.4 Kernel Function and Threads  (block, grid)

• The kernel is a function that is executed by each thread launched on the GPU
• The threads executing the kernel are organized into groups called blocks
• All threads in all blocks form the grid
• For example:
	• 256 threads form a block;
	• launching 500 blocks mean that we have launched a grid of 128,000 threads.


---
Special Variables To Identify Threads & Blocks

We have a set of special variables to identify the thread executing the code:
threadIdx.x - the id inside of a block
blockIdx.x - the id of the block in the grid
blockDim.x - the number of threads in the block
blockDim.x * blockIdx.x + threadIdx.x - a global unique id of a thread

• We can use these build-in variables inside a kernel function, which is a void function labelled with the __global__ qualifier:
![[Pasted image 20251107152949.png]]





# 3 First Example: Vector Addition


![[Pasted image 20251107152625.png]]

• Vector addition is one of the easiest data-parallel programs we can write
• It is the “Hello World” of GPU computing
• This is an example of an embarrassingly parallel problem where very little effort is required to split the problem into parallel tasks:
	• Every components of the result vector can be computed independently of the others.

## 3.1 C++ Code

• Allocate input and output vectors and fill input vectors with random data
• We use modern C++ to avoid having to deal with manual memory management (manually calling free)

![[Pasted image 20251107152711.png]]


---

• Here we use a traditional for-loop
![[Pasted image 20251107152815.png]]

• In each iteration we add up two components of the input vectors
• We store the result in the output vector
• Note that each iteration is independent of the others
• We show two versions:  one in modern C++ and one in C
![[Pasted image 20251107152822.png]]

## 3.2 CUDE Program

What is our strategy to turn this computation into a kernel that is executed in parallel by many threads?
• Idea: have one thread execute one iteration of the loop!


![[Pasted image 20251107153022.png]]

```
dim3 grid1((numElements + threadsPerBlock - 1) / threadsPerBlock);
dim3 block1(threadsPerBlock);
vecAdd<<<grid1, block1>>>(d_a, d_b, d_tmp, numElements);
```


1  `dim3 block1(threadsPerBlock);`
这一行定义了 **线程块的大小**。
- `dim3` 是 CUDA 的一个三维结构体，用来描述维度（x, y, z）。
- 这里只设置 `x` 维度，即：
    `block1.x = threadsPerBlock;   // 每个线程块里有多少个线程 block1.y = block1.z = 1;      // 默认值`

例如 表示每个线程块（block）有 256 个线程（编号 0～255）。


2
dim3 grid1((numElements + threadsPerBlock - 1) / threadsPerBlock);

这一行定义了 网格（grid）里线程块的数量。
网格是由多个 block 组成的；block 由多个线程组成。
我们希望能覆盖所有 numElements 个元素（每个线程处理一个元素）。
因此计算公式是： `(numElements + threadsPerBlock - 1) / threadsPerBlock`
- 是 C/C++ 实现上取整的写法（防止丢掉最后几个元素）。

```
numElements = 1000
threadsPerBlock = 256

```

→ (1000 + 255) / 256 = 4

```
grid1.x = 4;   // 共 4 个 block
block1.x = 256 // 每个 block 有 256 个线程

```

总线程数 = `4 × 256 = 1024`  
前 1000 个线程会执行有效操作，最后 24 个线程会在 kernel 内被忽略（通过 `if (i < numElements)` 判断）。


3
vecAdd<<<grid1, block1>>>(d_a, d_b, d_tmp, numElements);

这行代码就是 启动 kernel（核函数调用）。

它在 GPU 上并行执行如下函数：

### 3.2.1 含义：

- `<<<grid1, block1>>>`  
    表示启动：
    
    - `grid1.x` 个线程块
        
    - 每个块内 `block1.x` 个线程  
        所以总线程数约为 `grid1.x * block1.x`。
        
- 每个线程在 GPU 上执行一次 `vecAdd()` 函数体。
    
- 在 kernel 内部，这些特殊变量自动可用：

```
blockIdx.x    // 当前线程块在网格中的索引
threadIdx.x   // 当前线程在线程块中的索引
blockDim.x    // 每个块中的线程数 (threadsPerBlock)
```

所以每个线程计算自己的全局索引：
```
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < numElements) c[i] = a[i] + b[i];
```

每个线程各自处理一个 i，实现完全并行的向量加法。



4

|概念|作用|类比|
|---|---|---|
|**grid**|整个任务的划分|全厂所有工人|
|**block**|一组线程（一起工作的工人组）|每个车间|
|**thread**|实际干活的单个线程|每个工人|

## 3.3 First Full CUDA Example: Vector Addition

• Host code, unchanged to prior CPU-only version
• A, B, C, are now prefixed with h_ to indicate that these are stored in host memory

![[Pasted image 20251107153112.png]]

---

• Allocate device memory for input vectors A and B & output vector C
• Error checking is very helpful for detecting errors early!

![[Pasted image 20251107153350.png]]

---

• Copy input data from the host to the device
![[Pasted image 20251107153406.png]]


• launch kernel with one thread per vector component organized into blocks of 256 threads
![[Pasted image 20251107153419.png]]


• Copy output data from the device to the host
![[Pasted image 20251107153438.png]]


• deallocate device memory
![[Pasted image 20251107153444.png]]



# 4 Challenges To Find the Right Launch Configuration

![[Pasted image 20251107153738.png]]



![[Pasted image 20251107153751.png]]


1 
blockPerGrid
- 网格是由多个 block 组成的；block 由多个线程组成。
- 我们希望能覆盖所有 `numElements` 个元素（每个线程处理一个元素）。
- 因此计算公式是： `(numElements) / threadsPerBlock`  

`(numElements + threadsPerBlock - 1) / threadsPerBlock`  
是 C/C++ 实现上取整的写法（防止丢掉最后几个元素）。

# 5 Compilation

• CUDA files have the extension .cu and can mix host and device code
• They get compiled using Nvidia’s nvcc compiler: `nvcc -std=c++20 vecAdd.cu -o vecAdd`
• nvcc splits the code into device and host code
• the host code is compiled by a standard c++ compiler (GCC or Clang)
• the device code is compiled using Nvidia’s GPU compiler
• Everything is linked together into a single executable binary

Compilation Process
![[Pasted image 20251107154020.png]]


# 6 Execution of Kernels on the GPU

![[Pasted image 20251107154045.png]]


• All threads in one block are guaranteed to be scheduled for simultaneously execution on the same Streaming Multiprocessors (SM)
• A SM can execute multiple blocks at the same time
• All threads in one block share the resources of a single SM:
• 65.536 32-bit registers
• 96KB L1 Data Cache / Shared Memory
• On current GPUs, a block may contain up to 1024 threads.


![[Pasted image 20251107154105.png]]


## 6.1 Warp Execution Model

• As discussed last week, threads (from within one block) are grouped into groups of 32, called warps, for execution
• For our block of 256 threads, the following 8 warps are formed:
`[0, …, 31], [32, …, 63], [64, …, 95], [96, …, 127], [128, …, 159], [160, …, 191], [192, …, 223], [224, …, 255]`
• At each clock cycle one warp is selected and threads from within that thread that perform the same instruction are grouped and one such sub-warp group executes together

## 6.2 Independent Block Execution


• Blocks execute independently, possibly on different SMs
• It is possible to launch up to 2.147.483.647 (= 231 - 1) blocks!
• Of course, we don’t have enough physical resources to have that many blocks simultaneously active on the GPU!
• As blocks are completely independent, one block can also execute after another has already completely finished and before another even has started executing.



# 7 Performance Implications of Choosing a Launch Configuration

• Choosing a good or bad launch configuration can make a big difference for
performance
• The Occupancy Calculator (part of the NVIDIA Nsight Compute tool) can help
• You enter the chosen launch configuration and resource usage and it tells how well the GPU resources are occupied
• The number of registers used by a kernel can be found out using the —ptxas-options=-v nvcc command line option



# 8 Multidimensional Grids

• The organization of GPU threads can be one-, two-, or three-dimensional
• This can help when processing two- or three-dimensional data, such as images, matrices, or vectors
![[Pasted image 20251107154417.png]]

• For the warp-based execution, the multi-dimensional thread organization is flattened to form the warps

![[Pasted image 20251107154428.png]]

