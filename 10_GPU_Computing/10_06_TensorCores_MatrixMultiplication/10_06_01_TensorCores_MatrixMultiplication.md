
# 1 Tensor Cores

![](image/Pasted%20image%2020260215171804.png)

- Tensor Cores perform efficient matrix operations on small dense matrices
- Tensor Cores where first introduced in the Volta GPU architecture
- Improved Tensor Core Designs have featured since in the Turning, Ampere, Ada Lovelace, and Hopper Architectures


![](image/Pasted%20image%2020260215171920.png)


# 2 Tensore Cores - Hardware

First we look at the various hardware implementations over the GPU generations
We will also look at the hardware instructions that are added for each generation
Later we will look at how to program the Tensor Cores from CUDA


## 2.1 Tensor Cores 1st Generation (Volta)


- A 1st generation tensor core computes a MMA: matrix multiplication add
- The inputs A and B are FP16 matrices. The matrices C and D are FP16 or FP32 
- Each Tensor Core performs 64 FP16 FMA operations per clock cycle 8 Tensor Cores per SM = 512 FMA per clock cycle = 1024 FP16 FLOP per cycle
- 640 Tensor Cores deliver 125 TFLOPS with FP16 (vs. 15.7 TFLOPS Peak FP32 of the CUDA Cores)

![](image/Pasted%20image%2020260215172944.png)

Implementation in Hardware Based on Systolic Arrays
- Systolic Arrays are an old idea: arrange processing elements in a tight network and push data through it
- First used in the Colossus computer during the second world war


![](image/Pasted%20image%2020260215173406.png)



## 2.2 Tensor Cores 2nd Generation (Turing)

The inputs A and B can now be matrices with FP16,INT8,or INT4
The matrices C and D are FP16 or FP32
- 8 Tensor Cores per SM perform: 
    - 1024 FP16 ops per cycle
    - or 2048 INT8 ops per cycle
    - or 4096 INT4 ops per cycle
576 Tensor Cores deliver 130.5 TFLOPS with FP16 (vs. 15.3 TFLOPS Peak FP32)


![](image/Pasted%20image%2020260215174232.png)


![](image/Pasted%20image%2020260215174249.png)


## 2.3 Tensor Cores 3rd Generation (Ampere)


- A 3rd generation tensor core computes D = A x B + C. where A is 8x8 matrix, B is a 8x4 matrix
- A and B can now be in FP16,INT8,INT4,TF32,BF16 and FP64
-  Each Tensor Core performs 256 FP16 FMA ops per cycle 4 Tensor Cores per SM perform 2048 FP16 ops per cycle
- 576 Tensor Cores deliver 312 TFLOPS with FP16 (vs. 19.5 TFLOPS Peak FP32 of the CUDA Cores)


![](image/Pasted%20image%2020260215174418.png)

Comparison of Tensor Core capabilities between 1st (V100) and 3rd (A100) generation
![](image/Pasted%20image%2020260215174641.png)

Fine-Grained Structured Sparsity
- Tensor Cores exploit special form of sparsity elements
- In every four elements two are allowed to be non-zero otherwise represent this not allowed
- skips the compute on entries that have zero values resulting in a doubled performance

![](image/Pasted%20image%2020260215174830.png)

![](image/Pasted%20image%2020260215174845.png)


## 2.4 Tensor Cores 4th Generation (Hooper & Ada Lovelace)

- Double the raw compute power compared to Ampere:  D = A x B  + C， where A is a 8x16 matrix & B is a 16x4 matrix
- Each Tensor Core performs 512 FP16 FMA ops per cycle， 4 Tensor Cores per SM perform 4096 FP16 ops per cycle
- New FP8 data format
- 528 Tensor Cores deliver 989.4 TFLOPS with FP16 (vs. 66.9 TFLOPS Peak FP32 of the CUDA Cores)



SM90_64x16x16_F32F16F16_SS
- This instruction performs a 64x16 and 16x16 MMA (but wider range of sizes supported) 
- Is executed by multiple Tensor Cores 
- Is executed asynchronously
- It loads its inputs directly from shared memory (and not from registers as the instructions before)

![](image/Pasted%20image%2020260215175403.png)

## 2.5 Tensor Cores Compared Across Generations

 Number of FP16 FMA  hardware units per TC ： 每张量核心每周期 FP16 FMA 数量

FMA 是 Fused Multiply-Add 的缩写，中文意为融合乘加运算
计算公式为 D = A * B + C

![](image/Pasted%20image%2020260215175438.png)


# 3 Tensore Cores - Software

There are three different ways how to use the Tensor Cores:
- Directly in CUDA using the Warp Level Matrix-Multiply-Accumulate (WMMA) API
- Using NVIDIA CUTLASS (CUDA Templates for Linear Algebra Subroutines) 
- Using NVIDIA cuBLAS (CUDA Basic Linear Algebra Subroutines)


## 3.1 WMMA API

his API is used from within a CUDA kernel
A warp is cooperatively executing these operations
The API consists of a datatype and four functions:
![](image/Pasted%20image%2020260215175705.png)

### 3.1.1 fragment
`template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;`
- A fragment represents a section of a matrix that is distributed across all threads in the warp
- On most architectures the matrix is stored distributed across many registers 
- Use is either matrix_a, matrix_b, or accumulator
- m,n,k are the the matrix dimensions and must match the data type T
- Layout is either row_major or col_major

这段代码是 **NVIDIA CUDA C++ 的 WMMA (Warp Matrix Multiply-Accumulate) API** 中用于定义张量核心操作的核心数据结构。让我详细解释这个 `fragment` 类的各个参数及其含义：

```cpp
template<typename Use, int m, int n, int k, typename T, typename Layout=void> 
class fragment;
```

`Use` - 片段用途**
指定这个片段在矩阵乘加操作中扮演的角色：
- **`matrix_a`**：表示这是左乘矩阵（A矩阵）的片段
- **`matrix_b`**：表示这是右乘矩阵（B矩阵）的片段
- **`accumulator`**：表示这是累加器（C/D矩阵）的片段

 `m, n, k` - 矩阵维度**
这些维度必须与使用的张量核心支持的尺寸匹配：
- **`m`**：A矩阵的行数 / C矩阵的行数
- **`n`**：B矩阵的列数 / C矩阵的列数
- **`k`**：A矩阵的列数 / B矩阵的行数（内维）

常见维度组合（取决于T类型和GPU架构）：

| 架构 | 支持的维度 (m×n×k) | 数据类型 |
|------|-------------------|----------|
| Volta/Turing/Ampere | 16×16×16 | FP16 |
| | 32×8×16 | FP16 |
| | 8×32×16 | FP16 |
| Ampere+ | 16×16×8 | TF32 |
| | 16×16×4 | INT8 |
| Hopper+ | 64×64×16 | FP16 |
| | 32×32×8 | FP32 |

`T` - 数据类型**
指定存储在片段中的元素数据类型：
- **`__half` / `half`**：16位浮点数（FP16）
- **`__nv_bfloat16`**：16位脑浮点数（BF16）
- **`float`**：32位浮点数（FP32）
- **`int`**：32位整数（用于INT8/INT4量化）
- **`double`**：64位浮点数（FP64，部分架构支持）

`Layout` - 内存布局（可选）**
指定矩阵在内存中的存储顺序：
- **`row_major`**：行主序（行优先）
- **`col_major`**：列主序（列优先）
- **`void`**：默认值，表示布局由编译器根据上下文推导


### 3.1.2 **寄存器分布特性**

```cpp
// 示例：每个片段分布在warp内所有线程的寄存器中
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half> a_frag;

// 这个16x16的矩阵片段并不是完整存储在一个线程中
// 而是分布在warp内32个线程的寄存器中
// 每个线程持有矩阵的一部分元素
```

**存储方式：**
- **跨线程分布**：矩阵的每个元素不是完整存储在一个线程中，而是分布在warp的所有32个线程间
- **寄存器利用**：每个线程使用多个寄存器存储自己负责的矩阵元素
- **隐式共享**：WMMA API自动处理线程间的数据共享，对开发者透明

**内存到fragment的加载**

```cpp
// 从全局内存加载数据到fragment
nvcuda::wmma::load_matrix_sync(a_frag, d_A, lda);

// 执行矩阵乘法累加
nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// 将结果存储回全局内存
nvcuda::wmma::store_matrix_sync(d_C, c_frag, ldc, 
                                 nvcuda::wmma::mem_row_major);
```



### 3.1.3 **为什么使用fragment？**

**性能优势**
1. **寄存器直接操作**：避免L1/L2缓存延迟
2. **warp级协同**：32个线程协同工作，最大化张量核心利用率
3. **隐式同步**：WMMA API自动处理线程同步

**抽象层次**
fragment提供了对张量核心的高级抽象：
- 开发者不需要关心底层寄存器的具体分配
- 不需要手动处理线程间的数据交换
- 专注于算法实现而非硬件细节

**跨架构兼容性**
相同的WMMA代码可以在不同代际的NVIDIA GPU上运行：
- 老架构使用CUDA核心回退
- 新架构自动利用张量核心加速


### 3.1.4 **fragment的局限性**

1. **固定维度**：维度必须匹配张量核心支持的模式
2. **warp内通信**：只能在一个warp（32线程）内使用
3. **数据类型限制**：不是所有数据类型都支持
4. **不能直接访问**：fragment中的元素不能像数组那样直接随机访问

```cpp
// ❌ 错误：不能直接索引fragment
float val = c_frag[0];  

// ✅ 正确：必须使用wmma API访问
wmma::store_matrix_sync(d_C, c_frag, 16, wmma::mem_row_major);
```

fragment是CUDA WMMA API的核心，它使得开发者能够高效地利用NVIDIA GPU的张量核心进行矩阵计算，而无需深入了解底层的硬件细节。


### 3.1.5 Supported Matrix Sizes and Element Types

Which sizes and data types are supported depends on the hardware device
m,n,k and T in the fragment declaration must match one of the table entries


```
__half =' FP12, float =' FP32, double =' FP64, unsigned/signed char = INT8, precision:)u4 =' INT4, precision:)b1 =' 1 bit,
__nv_bfloat16 =' alternative 16-bit float format, precision:)t32 =' alternative 19-bit float format
```

![](image/Pasted%20image%2020260215181155.png)

### 3.1.6 Load, Store, Fill Fragments


`void load_matrix_sync(fragment<.."> &a, const T* mptr, unsigned ldm, layout_t layout);`
Waits for all threads of the warp, then loads the matrix fragments from memory. Must be called by all threads in the warp. The values of mptr, ldm, layout and all template parameters for a must be the same for all threads in the warp.

`void store_matrix_sync(T* mptr, const fragment<.."> &a, unsigned ldm, layout_t layout);`
Waits for all threads of the warp, then stores the matrix fragments to memory. Must be called by all threads in the warp. The values of mptr, ldm, layout and all template parameters for a must be the same for all threads in the warp.


`void fill_fragment(fragment<.."> &a, const T& v);`
Usually, called by all threads in a warp with a common value for v.


### 3.1.7 Matrix-Multiply-Accumulate

```c++
void mma_sync(fragment<.."> &d, const fragment<.."> &a, const fragment<.."> &b, const fragment<.."> &c, bool satf=false);
```


Waits for all threads of the warp, then performs the warp-synchronous matrix multiply-accumulate: D=A*B+C
C=A*B+C is also allowed.
The function must be called by all threads in the warp.
If satf (saturate to finite value) is true, the elements in the accumulator will be prevented to overflow


### 3.1.8 Example of 16x16x16 MMA


The code performs a matrix multiplication of a 16x16 FP16 matrix A with another 16x16 FP16 matrix B in a single warp.
The result is stored in the FP32 matrix C

**完整使用示例**

```cpp
#include <mma.h>
using namespace nvcuda;

// 定义一个16x16x16的FP16矩阵乘法
void wmma_example() {
    // 声明片段
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // 初始化累加器片段为零
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 加载矩阵到片段
    wmma::load_matrix_sync(a_frag, d_A, 16);  // d_A是16x16的half矩阵，leading dimension=16
    wmma::load_matrix_sync(b_frag, d_B, 16);  // d_B是16x16的half矩阵
    
    // 执行矩阵乘加：C = A * B + C
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // 存储结果
    wmma::store_matrix_sync(d_C, c_frag, 16, wmma::mem_row_major);
}


#include <mma.h>
using namespace nvcuda;
__global__ void wmma_ker(half *a, half *b, float *c) {
   /, Declare the fragments
   wmma:)fragment<wmma:)matrix_a, 16, 16, 16, half, wmma:)col_major> a_frag;
   wmma:)fragment<wmma:)matrix_b, 16, 16, 16, half, wmma:)row_major> b_frag;
   wmma:)fragment<wmma:)accumulator, 16, 16, 16, float> c_frag;
   /, Initialize the output to zero
   wmma:)fill_fragment(c_frag, 0.0f);
   /, Load the inputs
   wmma:)load_matrix_sync(a_frag, a, 16);
   wmma:)load_matrix_sync(b_frag, b, 16);
   /, Perform the matrix multiplication
   wmma:)mma_sync(c_frag, a_frag, b_frag, c_frag);
   /, Store the output
   wmma:)store_matrix_sync(c, c_frag, 16, wmma:)mem_row_major);
}
```



## 3.2 CUTLASS

- CUDA C++ Template Library for writing fast Matrix Multiply Kernels
- Why is matrix multiplication such a big deal right now? =. Deep Learning 
- Is it not sufficient to just call a pre-implemented library implementation?
    - Yes, often it is, but we often want to specialize the computation to a particular situation, e.g., choosing data formats & layout, fusing in additional computations, optimize for a specific input size, or hardware architecture, etc ...
    - Also: somebody has to write the library implementation


### 3.2.1 Efficient Matrix Multiplication On GPUs


For M x N x K = 8192 x 128 x 8192 the arithmetic intensity is 124.1 FLOPS/byte
For M x N x K = 8192 x 8192 x 8192 the arithmetic intensity is 2730 FLOPS/byte
=>. larger matrix multiplications have higher arithmetic intensity only from a certain minimal size can we hope to exploit the full FLOPS of the GPU hardware architecture (see roofline model)


![](image/Pasted%20image%2020260215182120.png)

---
1 Inner -> outer product
![](image/Pasted%20image%2020260215182316.png)

The standard formulation computes the inner product (dot product) and requires reloading rows and columns from A and B over and over again, or holding them for a long time in the cache / shared memory.
By permuting the loop nests, we now compute the outer products of a column of A and a row of B once, never using the column and row again.

---

2 Tiling
![](image/Pasted%20image%2020260215182430.png)

We partition matrix C (and with it A and B) to guarantee that the tile of C fits into the fast on-chip memory.

---

3 Accumulating dot products -> accumulating matrix products: to exploit hardware hierarchy

To make use of the hardware resources available, we move to accumulate the product of matrices rather than the product of vectors.
We want to exploit the hierarchical hardware structure of the GPU with its blocks, warps, and threads and global, shared memory, and registers.
![](image/Pasted%20image%2020260215182537.png)


---

4 Block Tile
Each block computes a part of the output matrix by loading blocks of the A and B matrix, multiply them and accumulate the result in C.
This sub-matrix multiplication is performed by the warps of the block in shared memory.

![](image/Pasted%20image%2020260215182654.png)

---

Warp Tile
Each warp computes a sequence of accumulated matrix products, by iteratively loading submatrices from shared memory to registers, computing an accumulated outer product cooperatively by the threads of the warp (left) or by the Tensor Cores (right).

![](image/Pasted%20image%2020260215183146.png)

---

4 Software Pipelining
A high demand of registers and shared memory limits the amount of threads we can launch to hide memory latency. We can employ software pipelining to hide the data movement latency.

![](image/Pasted%20image%2020260215183335.png)



### 3.2.2 Matrix Multiplication With CUTLASS

CUTLASS is a library that helps to efficiently navigate the hierarchal tiling model
It helps to tile memory and efficiently move data in the memory hierarchy, by purely configuring the matrix multiply
It also allows to fuse element wise operations into the matrix-multiply kernel
Example of CUTLASS 1.0 on the right

![](image/Pasted%20image%2020260215181740.png)


### 3.2.3 Efficient Matrix Multiplication on Hopper GPUs

![](image/Pasted%20image%2020260215182016.png)



## 3.3 CuBLAS


- cuBLAS is NVIDIAs implementation of the BLAS standard
- This is a well established standard of linear algebra operations
- Can fairly easily be called from host code without any CUDA or kernel programming knowledge
- Library is highly optimized and gives the best Matrix Multiply performance
- But, interface is inflexible, e.g., in data types and fusion of additional operations into the computation is not possible

![](image/Pasted%20image%2020260215180121.png)

cuBLAS chooses between many different kernels optimized for  different matrix sizes and hardware
Theoretical performance limit  for FP16: 312 TFLOPS

