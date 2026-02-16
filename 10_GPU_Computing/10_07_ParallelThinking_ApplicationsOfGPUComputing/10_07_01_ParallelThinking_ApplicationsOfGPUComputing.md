
# 1 Parallelize Computations


There are usually three main reasons why people pursue parallel computing:
To solve a given problem in less time
To solve bigger problems within a given amount of time
To solve a given problem in a given amount of time better  (e.g., by producing a more accurate solution)
Parallel computing comes with the cost of increased software complexity!


## 1.1 Speedup

How to we measure the performance benefits of parallel code? Speedup = t(1) / t(N)
where: t(1) is the computational time when using 1 processor  t(N) is the computational time when using N processors.

We can also apply this to CPU vs. GPU: SpeedupCPUvsGPU = t(CPU) / t(GPU)

The speedup is the relative performance of two systems solving the same problem.


---

What Is a Good Speedup?
When we increase the number of processors participating in the computation, we would like to see a linear speedup: t(1) / t(N) = N
This is usually unrealistic, due to algorithmic or hardware characteristics
Particularly, on the GPU, we know that launching one thread (and using a single processor core) is a terrible idea. It is not appropriate to use this as the baseline for speedup comparisons!
Sometime, we achieve speedups of >N these are called super-linear and come from exploiting hardware characteristics, e.g., the larger cache when using N cores.


---

What Are the Limits of Speedup?
What happens when we increase the number of processors?  What is the highest speedup we can achieve?
In 1967, Gene Amdahl pointed out that "the overall performance improvement gained by parallelizing a single part of a system is limited by the fraction of time that the improved part is actually used".


## 1.2 Amdahl's Law and Strong Scaling


Speedup = 1 / (s + p/N)
where s is the proportion of time spent on the serial part, 
p is the proportion of time spent on the parallel part,  N is the number of processors.

Amdahl's law tells us that if we can parallelize 50% of the overall runtime, we can never be faster than 2x.
For 90% the maximum speedup is 10x.
For 95% the maximum speedup is 20x.
Amdahl's law makes an observation for strong scaling, i.e., when we keep the problem size constant and increase  the number of cores.

![](image/Pasted%20image%2020260215190501.png)


## 1.3 Gustafson's Law and Weak Scaling

In 1988, John L. Gustafson pointed out, that "One does not take a fixed-size problem and run it on various numbers of processors except when doing academic research; in practice, the problem size scales with the number of processors. When given a more powerful processor, the problem generally expands to make use of the increased facilities."

1988年，约翰·L·古斯塔夫森指出："人们并不会拿一个固定规模的问题去在不同数量的处理器上运行，除非是在做学术研究；在实践中，问题规模会随着处理器数量的增加而扩展。当获得性能更强的处理器时，问题通常会相应扩大，以充分利用增强的计算能力。"

This motivated Gustafson's law:   Speedup = s + p x N
based on the observation that the parallel part of the code scales with the number of processors and the serial part does not increase.

where s is the proportion of time spent on the serial part, 
p is the proportion of time spent on the parallel part,  N is the number of processors.
s 表示串行部分所占的时间比例，
p 表示并行部分所占的时间比例，
N 表示处理器的数量。

With Gustafson's law there is no upper limit for the speedup.
The serial part determines just the slope of a linear function.
We call it weak scaling, when the problem size increases with the number of processes.

![](image/Pasted%20image%2020260215190658.png)


# 2 Parallel Programming

How do we parallelize a problem, i.e., come up with a solution that exploits the parallelism available in the hardware?
Generally we can think in three conceptual steps:
- Algorithm selection 
- Problem decomposition 
- Performance optimizations


## 2.1 Parallel Algorithms

Conclusions
- Often the selection of parallel algorithms isn't straightforward
- Both algorithmic, as well as hardware aspects have to be kept in mind
- Developing good parallel algorithms for a specific hardware, such as GPUs,  is challenging and there are regularly research papers published presenting new parallel and GPU-friendly algorithms

---

Consider Tradeoffs of Algorithmic Complexity vs. Hardware Characteristics

Most algorithms are formulated as serial (or sequential) algorithms with the assumption that a single processor executing it
Special parallel algorithms must be developed that describe how the problem is solved by many cooperating processes.
Not all problems have (efficient) parallel algorithms
dependency resolution canheparal
Parallelism (performing multiple things at the same time) should not be confused with concurrency (which is the possibility, but not necessity, that multiple things happen at the same time).

大多数算法被设计为串行（或顺序）算法，其前提是假设由单个处理器来执行。
必须开发专门的并行算法，用以描述如何通过多个协作进程来解决问题。
并非所有问题都拥有（高效的）并行算法。依赖关系的解决可能会限制并行性。
并行性（同时执行多个任务）不应与并发性（多个任务可能、但不一定同时发生）混淆。


### 2.1.1 Example Problem: Scan


For examples of parallel algorithms, we look at the prefix sum, or scan, computation
The inclusive scan computes the following output array:   
 - given a binary associative operator  and inout array
 - A simple sequential algorithm  performs N-1 additions:

![](image/Pasted%20image%2020260215191047.png)


---

First idea: launch many threads, each performing a reduction.
Problem: the thread computing the reduction for the last element  will have basically the same runtime as the sequential scan. But we  would actually perform O(N2) additions instead of O(N)!
第一种思路： 启动多个线程，每个线程独立执行一个规约计算。
问题： 计算最后一个元素的线程，其运行时间基本上与串行扫描相同。而且，这样做实际上会执行 O(N²) 次加法，而不是 O(N) 次！

Second idea: Let's investigate the reduction trees and attempt to share partial sums across the different reduction trees
让我们研究一下规约树的结构，并尝试在不同的规约树之间共享部分和。


---

### 2.1.2 Kogge-Stone Scan Algorithm
• Turn input elements xi into output elements yi, implemented in-place
• Perform multiple iterations, after iteration k element xi contains the sum of 2k-1 elements before it.

![](image/Pasted%20image%2020260215191250.png)


• 将输入元素 xi 转换为输出元素 yi，采用原地（in-place）实现
• 执行多次迭代，在第 k 次迭代后，元素 xi 包含了它之前 2ᵏ⁻¹ 个元素的总和


**算法说明：**

Kogge-Stone 扫描是一种经典的并行前缀和算法，具有以下特点：

1. **工作原理**：
   - 迭代 1：每个元素加上它前面 1 个元素
   - 迭代 2：每个元素加上它前面 2 个元素
   - 迭代 3：每个元素加上它前面 4 个元素
   - 以此类推

2. **时间复杂度**：O(log N) 步，每步 O(N) 操作

3. **优点**：
   - 计算深度浅（log₂N 级）
   - 适合 GPU 等并行架构
   - 规则的数据访问模式

4. **缺点**：
   - 需要较多的线程/处理器（O(N log N) 总操作数）
   - 对长向量的实现可能需要大量共享内存

Kogge-Stone 算法是 GPU 编程中实现高效并行扫描的基础，特别适用于 CUDA 等并行计算平台。

![](image/Pasted%20image%2020260215191334.png)

![](image/Pasted%20image%2020260215191406.png)



- 这里我们执行了多少次加法  
  \[  \sum_{步长} (N - \text{步长}), \text{ 步长为 } 1, 2, 4, \dots N/2(\log_2 N \text{ 项})\]
  \[  = N \cdot \log_2(N) - (N - 1)\]

- 而串行算法只执行了 \((N - 1)\) 次加法。

- 我们在并行算法中执行了相当多的额外工作！这说明该算法并非**工作高效型**。

- 我们执行了 \((N \cdot \log_2(N))/P\) 次并行迭代，而串行版本需要执行 \(N\) 次迭代。

- 对于 \(N=P=512\)，加速比为 \(512/(512 \cdot \log_2(512)/512) = 56.9\) 倍。

- 对于 \(N=1024\)，\(P=32\)，加速比为 \(1024/(1024 \cdot \log_2(1024)/32) = 3.2\) 倍。

![](image/Pasted%20image%2020260215191634.png)

---

**关键点总结：**
- Kogge-Stone 算法虽然实现了并行化，但付出了额外计算量的代价
- 加速效果取决于问题规模 \(N\) 与处理器数量 \(P\) 的比例
- 当处理器数量与问题规模相当时，加速效果显著；当处理器相对较少时，加速效果有限




### 2.1.3 Brent-Kung Scan Algorithm


Can we improve the work efficiency,  i.e., do less extra work?
Basic idea: do a reduction and then build "reverse reduction trees" to distribute the remaining results



- 我们能否提高**工作效率**，即减少额外执行的计算量？
- **基本思想**：先执行一次规约（reduction），然后构建"反向规约树"来分发剩余的结果

---

**算法特点：**

Brent-Kung 扫描是一种在**工作高效性**和**并行度**之间取得平衡的并行前缀和算法：

1. **两阶段设计**：
   - **第一阶段（规约阶段）**：构建一棵规约树，计算部分和
   - **第二阶段（分发阶段）**：利用反向树，将部分和传播到所有位置

2. **时间复杂度**：约 2 log₂ N 步

3. **操作复杂度**：约 2N 次加法

4. **与 Kogge-Stone 对比**：
   - Kogge-Stone：O(N log N) 操作，log N 步
   - Brent-Kung：O(N) 操作，2 log N 步
   - **Brent-Kung 更工作高效**（操作量更少），但需要更多迭代步数

5. **适用场景**：
   - 当处理器资源有限时
   - 当计算量（而非并行深度）是主要瓶颈时
   - 在需要平衡资源利用率的系统中

Brent-Kung 算法体现了并行计算中的一个重要权衡：**用更多的时间步数换取更少的总体计算量**，从而提高工作效率。

![](image/Pasted%20image%2020260215192653.png)

![](image/Pasted%20image%2020260215191817.png)


这是对图片中 Brent-Kung 扫描算法 CUDA 内核代码的中文注释和解释：


| 方面 | 说明 |
|------|------|
| **每个块处理** | `SECTION_SIZE` 个元素（通常为 2 × blockDim.x） |
| **共享内存使用** | `XY[]` 数组存储当前块的所有元素 |
| **同步点** | 每个循环迭代后都需要 `__syncthreads()` |
| **索引计算** | `(threadIdx.x + 1)*2*stride - 1` 确保每个线程处理正确的元素 |
| **两阶段设计** | 向上规约 + 向下分发 |


**优点：**
- 工作高效（O(N) 操作）
- 共享内存使用优化
- 适用于各种块大小

**缺点：**
- 需要两阶段处理，延迟较高
- 需要仔细的同步管理
- 共享内存大小限制可能影响可处理的数据量

```c++
global void Brent Kung scan kernel(float *X, float *Y, unsigned int N) {
    // 在共享内存中分配数组，用于存储当前块处理的数据
    shared float XY[SECTION_SIZE];
    
    // 计算全局索引：每个块处理 2*blockDim.x 个元素
    // blockIdx.x: 块索引，blockDim.x: 块大小，threadIdx.x: 线程索引
    unsigned int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
    
    // 步骤1：将全局内存中的数据加载到共享内存
    // 每个线程加载两个元素（如果存在的话）
    if(i < N) 
        XY[threadIdx.x] = X[i];  // 加载第一个元素
    
    if(i + blockDim.x < N) 
        XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];  // 加载第二个元素
    
    // 步骤2：向上规约阶段（构建规约树）
    // 使用递增步长，计算部分和
    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();  // 同步，确保上一轮计算完成
        
        // 计算当前线程要处理的索引位置
        // 索引模式：1, 3, 5, 7, ...（基于步长）
        unsigned int index = (threadIdx.x + 1)*2*stride - 1;
        
        if(index < SECTION_SIZE) {
            // 将前一个步长位置的值加到当前位置
            XY[index] += XY[index - stride];
        }
    }
    
    // 步骤3：向下分发阶段（构建反向树）
    // 使用递减步长，将部分和传播到所有位置
    for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
        __syncthreads();
        
        unsigned int index = (threadIdx.x + 1)*stride*2 - 1;
        
        if(index + stride < SECTION_SIZE) {
            // 将当前值加到后一个步长位置
            XY[index + stride] += XY[index];
        }
    }
    
    // 步骤4：将结果从共享内存写回全局内存
    __syncthreads();
    
    if (i < N) 
        Y[i] = XY[threadIdx.x];  // 写回第一个元素
    
    if (i + blockDim.x < N) 
        Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];  // 写回第二个元素
}
```

---
### 2.1.4 流程图解

```
阶段1：向上规约（构建树）
步长=1:  [a] [b] [c] [d] [e] [f] [g] [h]
         ↓   ↓   ↓   ↓
         a  a+b  c  c+d  e  e+f  g  g+h

步长=2:  [a] [a+b] [c] [c+d] [e] [e+f] [g] [g+h]
                 ↓                   ↓
                 a+b+c+d             e+f+g+h

阶段2：向下分发（反向树）
步长=2:  [a] [a+b] [c] [a+b+c+d] [e] [e+f] [g] [e+f+g+h]
                 ↓                   ↓
         [a] [a+b] [a+b+c] [a+b+c+d] [e] [e+f] [e+f+g] [e+f+g+h]
```

让我们用更直观的方式，重新梳理一下 **Brent-Kung 扫描算法** 在 GPU 上的完整执行流程。


为了方便理解，我们设定以下参数：
- **SECTION_SIZE = 8** (即每个线程块处理 8 个数据)
- **blockDim.x = 4** (每个线程块有 4 个线程)
- 输入数据 `X` 为： `[1, 2, 3, 4, 5, 6, 7, 8]` (仅考虑一个块的情况)
- 共享内存数组 `XY` 大小为 8。

该算法分为四大步：
1.  **数据加载 (Load)**：将数据从全局内存读到共享内存。
2.  **向上规约 (Up-Sweep / Reduce)**：构建一棵树，计算部分和 (有点像计算总和的"前缀"版本)。
3.  **向下分发 (Down-Sweep)**：利用反向的树，将之前计算的部分和"分发"到空缺的位置，最终得到完整的前缀和。
4.  **结果写回 (Store)**：将结果从共享内存写回全局内存。

---


第 1 步：数据加载
- **操作**：每个线程（0-3）加载两个元素到共享内存 `XY`。
    - 线程 0：`XY[0] = 1`， `XY[4] = 5`
    - 线程 1：`XY[1] = 2`， `XY[5] = 6`
    - 线程 2：`XY[2] = 3`， `XY[6] = 7`
    - 线程 3：`XY[3] = 4`， `XY[7] = 8`
- **此时 `XY` 内容**：`[1, 2, 3, 4, 5, 6, 7, 8]`

第 2 步：向上规约 (Up-Sweep)
这是一个**逐级构建树**的过程。这一阶段结束后，树的根节点（最后一个元素）会存储所有元素的总和，而中间节点存储其"子树"的部分和。
- **迭代 1 (stride = 1)**：
    - **公式**：`index = (threadIdx.x + 1)*2*stride - 1`。当 `stride=1` 时，`index = 1, 3, 5, 7`。
    - **操作**：`XY[1] += XY[0]`; `XY[3] += XY[2]`; `XY[5] += XY[4]`; `XY[7] += XY[6]`。
    - **结果**：`[1, 3, 3, 7, 5, 11, 7, 15]`

- **迭代 2 (stride = 2)**：
    - `index = 3, 7`
    - **操作**：`XY[3] += XY[1]`; `XY[7] += XY[5]`。
    - **结果**：`[1, 3, 3, 10, 5, 11, 7, 26]`

- **迭代 3 (stride = 4)**：
    - `index = 7`
    - **操作**：`XY[7] += XY[3]`。
    - **结果**：`[1, 3, 3, 10, 5, 11, 7, 36]`

**此时 `XY` 内容**：`[1, 3, 3, 10, 5, 11, 7, 36]`
*注意：这里的值已经不是原始数据了，它们是特定区间的和。*

第 3 步：向下分发 (Down-Sweep)
这是**最关键的一步**。它的起点是上一步结束时的数组，终点是每个位置都变成该位置之前的**包含前缀和**。
- **前置操作**：先将最后一个元素（根节点）置为 0（或者在算法逻辑上视为"标识符"，用于启动传播）。在标准实现中，通常会利用上一步的最后一个元素来推导，但这段代码的索引方式稍有不同，我们需要结合代码逻辑看：
    - 代码中，向下循环是从 `SECTION_SIZE/4` 开始的。`SECTION_SIZE=8`，所以 `stride = 2` 开始。

- **迭代 1 (stride = 2)**：
    - **公式**：`index = (threadIdx.x + 1)*stride*2 - 1`。当 `stride=2` 时，`index = 3, 7`。
    - **操作**：`if(index + stride < SECTION_SIZE)`，即处理 `index=3` 和 `index=7` 的情况。
        - `XY[3 + 2] += XY[3]`  →  `XY[5] += XY[3]`
        - `XY[7 + 2]` 超出范围，不处理。
    - **计算**：`XY[5] = 11 + 10 = 21`
    - **结果**：`[1, 3, 3, 10, 5, 21, 7, 36]`

- **迭代 2 (stride = 1)**：
    - `index = 1, 3, 5, 7`
    - **操作**：
        - `XY[1 + 1] += XY[1]` → `XY[2] += XY[1]` → `3 + 3 = 6`
        - `XY[3 + 1] += XY[3]` → `XY[4] += XY[3]` → `5 + 10 = 15`
        - `XY[5 + 1] += XY[5]` → `XY[6] += XY[5]` → `7 + 21 = 28`
        - `XY[7 + 1]` 超出范围。
    - **结果**：`[1, 3, 6, 10, 15, 21, 28, 36]`

**此时 `XY` 内容**：`[1, 3, 6, 10, 15, 21, 28, 36]`
*这正是我们想要的包含前缀和的结果！*

第 4 步：结果写回
将 `XY` 中的数据写回全局内存 `Y` 即可。



```
原始数据: [1, 2, 3, 4, 5, 6, 7, 8]

第2步: 向上规约 (Up-Sweep)
1.  stride=1: [1, 3, 3, 7, 5, 11, 7, 15]  (两两求和)
2.  stride=2: [1, 3, 3, 10, 5, 11, 7, 26] (相隔2求和)
3.  stride=4: [1, 3, 3, 10, 5, 11, 7, 36] (相隔4求和)

第3步: 向下分发 (Down-Sweep)
1.  stride=2: [1, 3, 3, 10, 5, 21, 7, 36] (将10传递给5)
2.  stride=1: [1, 3, 6, 10, 15, 21, 28, 36] (将3传给3->6, 10传给5->15, 21传给7->28)

最终结果: [1, 1+2, 1+2+3, ..., 1+...+8]
```


---

![](image/Pasted%20image%2020260215191827.png)


- 我们执行了多少次加法：  
  **规约阶段**执行 \(N-1\) 次加法  
  **反向树阶段**执行 \(N-1-\log_2(N)\) 次加法  

- 总操作量是 \(O(N)\)，而之前（Kogge-Stone）是 \(O(N \cdot \log_2 N)\)！

- 我们获得了更好的**工作高效性**！

- 对于 \(N=1024\)，\(P=32\)，加速比约为 **16 倍**（相比之下 Kogge-Stone 只有 3.2 倍）

- 但如果所有线程并行执行，我们需要等待更长时间才能得到结果，因为**工作量最大的线程**需要先执行规约，然后等待反向树的传播

---

### 2.1.5 两种算法比较 

| 算法 | 总操作量 | 并行步数 | 工作高效性 | 适用场景 |
|------|----------|----------|------------|----------|
| Kogge-Stone | \(O(N \log N)\) | \(\log N\) | 低 | 线程资源丰富，追求最小延迟 |
| Brent-Kung | \(O(N)\) | \(2\log N\) | 高 | 计算资源有限，追求效率 |

Brent-Kung 算法体现了并行计算中的一个重要权衡：**以更多的时间步数换取更高的计算效率**，在资源受限的情况下尤为有价值。

Brent-Kung 算法通过**两棵树**的巧妙配合：
1.  **规约树**（向上）收集信息，产生部分和。
2.  **分发树**（向下）传播信息，将部分和"填空"到正确的位置。

相比Kogge-Stone算法（一步到位，但计算量大），Brent-Kung虽然需要更多的循环迭代（`2*logN` vs `logN`），但总的加法操作次数大幅减少（`O(N)` vs `O(NlogN)`），因此被称为**工作高效型**算法。


## 2.2 Problem Decomposition

Once a parallel algorithm has been found, we must decompose our problem into sub-problems that can be safely be solved in parallel.
There are two common strategies for decomposition:
- Input-centric: assign threads to process different input elements 
- Output-centric: assign threads to produce different output elements

![](image/Pasted%20image%2020260215191945.png)

### 2.2.1 Input-Centric Decompositions


Parallel Histogram 
Reduction
Output-centric decomposition would be problematic, as there are much more input elements than output elements, so there would be a lack of parallelism.


![](image/Pasted%20image%2020260215192941.png)

Example: Sparse Matrix Vector Multiplication
Each thread is assigned  non-zero elements in the  input and updates the  element in the output vector
![](image/Pasted%20image%2020260215210900.png)

## 2.3 Output-Centric Decompositions

Image Processing + MM
Image processing applications  such as the Sobel Filter 
edge detector
Matrix-Matrix-Multiplication
![](image/Pasted%20image%2020260215210942.png)



### 2.3.1 Examples of : Stencil Computations

stencil: 模板喷画

Stencil Computations are a common computational pattern
Each thread is computing  one element in the output (here green)  by performing a computation  over the neighborhood of  the corresponding element (here blue).
Stencils are a foundation to many  numerical methods for solving partial differential equations
![](image/Pasted%20image%2020260215211044.png)

---

Boundary Handling
Stencils have to think about how to handle accesses at the boundary where it is not easily possible to form a neighborhood
Different strategies exist that model different behavior.
This view, reinforces the output-centric  decomposition, as we only need to launch  threads for the elements we eventually  want to compute
![](image/Pasted%20image%2020260215211125.png)


---

Tiling
Elements next to each other have overlapping neighborhoods!
We can exploit this, by loading multiple neighborhoods into the shared memory. We call a collection of such neighborhoods a tile.
Besides exploiting this spacial locality  we can also exploit a temporal locality  in iterative stencil computations.
Many sophisticated tiling strategies  have been developed (also for GPUs)
![](image/Pasted%20image%2020260215211214.png)

![](image/Pasted%20image%2020260215211234.png)


# 3 Case Study: Medical Imaging


Magnetic resonance imaging (MRI) consist of two phases: 
1. acquisition: performing the scan to collect data
2. reconstruction: making an image out of the data
The reconstruction performs an iterative  process refining the current reconstructed  image
One crucial step in each iteration is to Computing FHd 

![](image/Pasted%20image%2020260215211325.png)

---
Computing FHd sequential code

Observation: plenty of data parallelism 
We first compute Mu (rMu & iMu)
Then in the inner loop, we compute  the contribution of one sample to  each pixel/voxel in the output image  FhD (rFhD & iFhD)
M = number of samples
N = number of pixels/voxels

cos and sin are relatively expensive

How do we turn this into performant CUDA code?

![](image/Pasted%20image%2020260215211444.png)


---

Determining the decomposition
![](image/Pasted%20image%2020260215211518.png)

Input-centric approach: all threads write into all output elements => use atomics 
How do we move towards an output-centric approach?

We need to parallelize over the inner loop!  Let's restructure our computation to make this happen.
![](image/Pasted%20image%2020260215211540.png)


![](image/Pasted%20image%2020260215211549.png)

---

Output-centric decomposition


![](image/Pasted%20image%2020260215211718.png)

Output-centric decomposition & Reduce global memory accesses
![](image/Pasted%20image%2020260215211749.png)



Use Constant Memory to reduce Global Memory accesses even further
We can make use of the small (64KB) constant memory.  To do so, we must divide the data into 64KB chunks  and launch a new kernel for each chunk.
![](image/Pasted%20image%2020260215211910.png)



![](image/Pasted%20image%2020260215211932.png)

---

With Better Memory Layout
Compared to the naive first  kernel implementation, we now:
- avoid using atomics reduced the number of  global memory accesses
- make use of constant memory with a cache-friendly memory layout 

These optimizations should account for about a 10x improvement in performance!

![](image/Pasted%20image%2020260215212056.png)



# 4 Examples of Generally Poor Fits for GPUs

Sequential tasks: Examples include recursive algorithms, certain dynamic programming problems, and some graph traversal algorithms.

Small datasets: The overhead of transferring data between the CPU and GPU, along with GPU initialization time, may outweigh any performance benefits.

Limited parallelism: Some algorithms have inherent constraints on the degree of parallelism that can be achieved.

Memory-bound problems: GPUs generally have less memory than CPUs, and memory bandwidth can become a limiting factor.

Fine-grained branching: GPUs perform best when threads execute similar control flow paths.

Low arithmetic intensity: If a problem has a low ratio of arithmetic operations to memory accesses, the GPU may not fully utilize its computational power.


顺序执行任务：例如递归算法、某些动态规划问题和部分图遍历算法。

小数据集：CPU 与 GPU 之间的数据传输开销以及 GPU 的初始化时间，可能会超过任何潜在的性能提升。

并行度有限：某些算法在可实现的并行程度上有固有的限制。

内存受限问题：与 CPU 相比，GPU 通常可用内存较少，且内存带宽可能成为限制因素。

细粒度分支：当不同线程执行的代码遵循相似的控制流时，GPU 的性能最佳。

低计算强度：如果问题的计算强度较低（即算术运算与内存访问的比率较低），GPU 可能无法有效利用其计算能力。

# 5 Examples of Good Fits for the GPU

- **Large-scale matrix and vector operations**: Common in machine learning, scientific computing, and image processing.
- **Fourier transforms**: Also widely used in machine learning, scientific computing, and image processing.
- **Monte Carlo simulations**: Used in finance, physics, and other fields to model complex systems.
- **Molecular dynamics simulations**: Applied in chemistry, biochemistry, and physics.
- **Computational fluid dynamics**: Used in engineering, physics, and related disciplines.
- **Convolutional neural networks and computer vision algorithms**: Core components of modern image recognition and analysis.
- **Big data analytics**: Includes tasks such as clustering, classification, and regression.

大规模矩阵和向量运算：常见于机器学习、科学计算和图像处理领域。

傅里叶变换：同样广泛应用于机器学习、科学计算和图像处理。

蒙特卡洛模拟：用于金融、物理学及其他领域，以模拟复杂系统。

分子动力学模拟：应用于化学、生物化学和物理学。

计算流体动力学：用于工程学、物理学及相关学科。

卷积神经网络与计算机视觉算法：现代图像识别与分析的核心组成部分。

大数据分析：包括聚类、分类、回归等任务。


---

From a metaphorical point of view, the gpu can be seen as a person lying on a bed of nails. The person lying on top is the data and in the base of each nail there is a processor, so the nail is actually an arrow pointing from processor to memory. All nails are in a regular pattern, like a grid. If the body is well spread, it feels good (performance is good), if the body only touches some spots of the nail bed, then the pain is bad (bad performance).

从隐喻的角度来看，GPU可以被视为一个躺在钉床上的人。躺在上面的人是数据，而每根钉子的底部都有一个处理器，因此这根钉子实际上是一支从处理器指向内存的箭。所有的钉子都按照规则的图案排列，就像网格一样。如果身体均匀地铺开，感觉就会很好（性能良好）；如果身体只接触钉床的某些点，那么疼痛就会很剧烈（性能糟糕）。

