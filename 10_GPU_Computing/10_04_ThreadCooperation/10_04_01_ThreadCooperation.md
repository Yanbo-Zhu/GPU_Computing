

# 1 Memory of Nvidia Tesla T4 GPU

• The entire GPU has access to 16 GB of DRAM memory
• L2 Cache of 6MB shared among all 72 SMs / 4,608 CUDA Cores
• L1 Cache / Shared Memory of 96KB shared among 64 CUDA Cores in one SM
	• Can be configured to be either controlled by hardware (“Cache”) or by the programmer in software (“Shared Memory”)
• 16,384 32bit registers shared among 16 CUDA Cores

![[Pasted image 20251112132843.png]]wrap[]


# 2 Memory Spaces 

• The GPU memory hardware architecture is reflected in CUDA memory spaces
• There are three main memory spaces:
	• global memory - all threads in the grid have access to the same global memory
	• shared memory - each block has shared memory visible to all threads in the block
	• local memory - each thread has their own private local memory
• There are two more read-only memory spaces: constant and texture memory


![[Pasted image 20251112133128.png]]


![[Pasted image 20251118224028.png]]

Pointer to global Memory 就是 points to the GPU

`float local_momory[LARGE_NUMBER]`    Some variables are placed in slower local memory: (which is part of the DRAM memory)

• Arrays for which it cannot determine that they are indexed with constant quantities,
• Large structures or arrays that would consume too much register space,  Therefor they are are placed in slower local memory, rather in registers
• Any variable if the kernel uses more registers than available (this is also known as register spilling)



# 3 Thread Cooperation

![[Pasted image 20251112133617.png]]


 First most straightforward approach:
• launch as many threads as data items;
• each thread processes one data item;
• and updates the appropriate counter

![[Pasted image 20251112133625.png]]

---

Read-Modify-Write Race Condition

• Updating the counter requires:
1. reading the old value;
2. adding one;
3. writing the new value back.
• Performing these read-modify-write operations concurrently leads to a race condition where the outcome of the computation depends on the order and relative timing of the operations.
• (A) and (B) are possible executions. (A) computes the correct result, but (B) not!

![[Pasted image 20251112134101.png]]

## 3.1 Atomic Operations To Avoid Data Race

• We can avoid the data race by using an atomic operation:
	• The read-modify-write sequence is performed as an undividable unit
• However, no specific order between the threads is enforced!
• In CUDA kernels, we can perform a number of simple arithmetic operations
atomically (add, sub, min, mix, inc, dec) as well as atomically exchanging two
values in memory and a atomic compare and swap (CAS).


![[Pasted image 20251112134148.png]]

`atomicAdd(&(histo[pos / 4]), 1)`
1: how much time it adds
2 `histo[pos / 4]` points to the memory address 

---

Latency and Throughput of Atomic Operations
• Accessing global memory takes hundreds of clock cycles
• To hide this latency, we learned that we switch to another thread
• Therefore, it is key to have many simultaneous memory accesses
• Unfortunately, this doesn’t work when we access the same memory location!
• To improve the throughput of atomic operations we can attempt to reduce the number of accesses to the same memory location

不总忘一个地址去加1了

## 3.2 Privatization

• Idea: each block first computes its own histogram 

only at the end are they combined into the final histogram

![[Pasted image 20251112134549.png]]


---

### 3.2.1 Privatization in global memory

• Using `__syncthreads()` we ensure that all threads within a block have reached a point before any continues
• In addition all global and shared memory accesses prior to `__syncthreads()` are visible to all threads in the block 
• All threads within the block must participate in the barrier otherwise we have a dead lock


![[Pasted image 20251118224205.png]]

### 3.2.2 Privatization in shared memory

We need synchronizations to coordinate the thread execution: 
has to wait all threads of all blocks 

![[Pasted image 20251118224401.png]]


## 3.3 `__syncthreads() `在 CUDA 中的作用解释：

![[Pasted image 20251112134706.png]]

`__syncthreads()` 是 CUDA C/C++ 中的一个**内置同步函数（barrier function）**，  
用于在同一个 **thread block**（线程块）内**同步所有线程的执行进度**。

当所有线程（0~3）都执行到 `__syncthreads()` 时，CUDA 运行时才允许它们 **一起继续** 执行后面的代码



CUDA 的执行模型中：
- 一个 **kernel** 会被分成若干个 **thread blocks**；
- 每个 block 中又有很多 **threads**；
- block 内的线程可以共享同一块 **shared memory**（共享内存）。

当程序执行到 `__syncthreads()` 时：
- **该 block 中所有线程** 都必须到达这个同步点；
- CUDA 会暂停已经到达的线程；    
- 直到 block 中的 **所有线程** 都到达 `__syncthreads()`，  
    然后所有线程才会一起继续往下执行。

这就像一个“集合点（barrier）”。


- **只能用于同一个 block 内的线程同步**，不同 block 之间无法用它同步。
- 必须保证所有线程都能执行到该语句，否则会造成 **死锁（deadlock）**。  
    比如： `if (threadIdx.x == 0) __syncthreads(); // ❌ 错误`   因为只有线程 0 执行了同步，而其他线程没执行，会导致永远等待。



# 4 Branch Divergence、 Warp Divergence

“降低分支发散”就是尽量让同一个 warp 内的 32 个线程走相同的执行路径。  
因为 GPU 按 warp 执行指令，一旦分支不同，warp 就必须**顺序执行多个分支**，性能大幅下降。

在 NVIDIA GPU 中，**线程是成组执行的**。
- 每 **32 个线程** 组成一个 **warp**（线程束）。
- 这 32 个线程**共享同一条指令流**（SIMT：Single Instruction, Multiple Threads）。
- 也就是说，warp 内所有线程 **必须同时执行同一条指令**，只是操作的数据不同。

分支发散（Branch Divergence）是什么意思？
当 warp 内的线程遇到 `if`、`else`、`switch` 这种分支语句时：

```
if (threadIdx.x < 16) {
    a[idx] *= 2;
} else {
    a[idx] *= 3;
}

```


对于一个 warp（32 个线程）：
- 前 16 个线程进入了 `if` 分支，
- 后 16 个线程进入了 `else` 分支。

GPU 不能真正地同时执行两个分支，于是会：
1. **先执行 `if` 分支**（只激活那 16 个线程，其他 16 个闲着），
2. 再 **执行 `else` 分支**（激活另外 16 个线程）。

这就意味着：

> 原本 32 个线程可以并行执行，现在只能分两次执行。  
> 性能相当于降低了一半。

这种情况就叫 **warp divergence（分支发散）**。


----
为什么要“降低分支发散”

因为它会：
- 降低并行度；
- 增加执行时间；
- 让某些线程处于 idle 状态（等待其他分支执行完）。

换句话说：
> GPU 只有在 warp 内所有线程执行同一条路径时，才能真正实现“满速”并行。



## 4.1 如何降低分支发散（几种方法）

1️⃣ 保证 warp 内线程处理**相似的数据或任务**

比如：

`int idx = threadIdx.x + blockIdx.x * blockDim.x; if (idx < N) {...}`

这是可以接受的分支，因为几乎所有 warp 的线程都会同样进入 if（除了最后一个 warp 可能部分越界）。

但如果写：

`if (in[idx] > 0) {...} else {...}`

而 `in[idx]` 随机分布在正负之间，那么 warp 内线程很可能一半走 if，一半走 else，就会严重发散。

👉 改进：  
用掩码或数学运算替代条件分支，比如：

`a[idx] *= (in[idx] > 0 ? 1 : -1);`

或使用 `fmaxf`, `fminf` 等内建函数避免显式 if-else。

---

2️⃣ 在算法层面重新安排任务

让同一个 warp 内的线程负责**同一种操作类型**。  
例如：
- 在 scan、reduce 等算法中，确保每个线程在同一个阶段执行相同的步长逻辑；
- 把特殊处理（例如边界处理）交给单独的 warp 或最后一个 block。

---

3️⃣ 使用 warp-level 原语

CUDA 提供了 **warp-level intrinsics**（如 `__shfl_xor_sync`、`__ballot_sync`），可以在 warp 内高效通信，而不需要 `if` 分支来区分不同线程。


# 5 thread Coarsening 

- 线程粗化（thread coarsening）：每个线程处理多个复数对；

• Privatization comes with an overhead: each block needs to write to the final histogram
• Idea: Reduce the number of blocks by increasing the work of a single thread

![[Pasted image 20251119140038.png]]


Interleaving is better because it loads the data closed compact data together than contiguous partitioning


## 5.1 thread Coarsening with contiguous partitioning

![[Pasted image 20251119140454.png]]

# 6 Memory Coalescing



![[Pasted image 20251119155258.png]]

When a warp executes an instruction that accesses global memory, the individual memory accesses are coalesced into one or more memory transactions
• Memory transactions are either 32-, 64-, or 128-byte wide
• It is particularly beneficial when the threads in the warp ==access consecutive memory locations==, as the least number of memory transfers must be performed


## 6.1 thread Coarsening with interleaved partitioning: 

Coarsening with interleaved partitioning for a better memory access pattern

![[Pasted image 20251119140547.png]]


# 7 Thread Cooperation: Parallel Reduction

这里的reduction不是数值相减， 而是用 parallel 减少计算次数 

![[Pasted image 20251119140632.png]]


• How do we parallelize reductions?
• When the reduction operation is associative, we are allowed to group the data arbitrarily and perform a tree-shaped reduction:
```
((((((3 max 1) max 7) max 0) max 4) max 1) max 6) max 3
=>
((3 max 1) max (7 max 0)) max ((4 max 1) max (6 max 3))
```
• When the reduction operator is also commutative, we are allowed to reorder the data which enables further optimizations.


## 7.1 Simple Reduction Kernel

• We start with a simple parallel reduction on the GPU
• We clearly need to cooperate among the threads to perform a tree-based reduction
• We know how to coordinate inside a block
• Let’s restrict ourselves for now to a kernel executed by a single block with B threads!
• Each thread processes two elements, therefore we can handle up to 2xB elements.

![[Pasted image 20251119140942.png]]

![[Pasted image 20251119141118.png]]


## 7.2 Minimizing control divergence / reduce divergence

降低分支发散：尽量让 warp 内线程走同一路；



![[Pasted image 20251119141137.png]]

• Remember that threads in warps execute their instructions together
• The way we have split the work among threads in each warp, so there are quickly some threads that are active and others that are not
• Instead we can better pack the active and not-active threads together
• This reduces divergent control flow


![[Pasted image 20251119141205.png]]


## 7.3 Minimizing global memory accesses

So far we have accumulated the intermediate results in global memory, let’s use shared memory instead!

利用共享内存：每个 block 内做分层前缀，减少全局访存；

![[Pasted image 20251119141235.png]]


## 7.4 Hierarchical reduction for arbitrary input length

• So far, we only launched a single block, restricting the maximal input length
• We did this, as we can not synchronize threads across blocks
• For a reduction of arbitrary input length we need to perform a hierarchical reduction:
• We split out input into segments, each reduced independently by a block;
• then we reduce the outputs further.
• The final reduction happens on the host, in a separate kernel, or using atomics.

![[Pasted image 20251119141332.png]]

![[Pasted image 20251119141353.png]]


## 7.5 Thread coarsening for reduced overhead

• Currently, 1/2 threads are only active for loading two elements from global memory and storing their sum in shared memory
• This is very wasteful!
• Idea: increase the work for a single thread!
• This optimization is called thread coarsening (and we have seen it before in the histogram as well)

![[Pasted image 20251119141423.png]]


![[Pasted image 20251119141436.png]]


