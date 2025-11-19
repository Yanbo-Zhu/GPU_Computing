

# 1 Memory of Nvidia Tesla T4 GPU

• The entire GPU has access to 16 GB of DRAM memory
• L2 Cache of 6MB shared among all 72 SMs / 4,608 CUDA Cores
• L1 Cache / Shared Memory of 96KB shared among 64 CUDA Cores in one SM
	• Can be configured to be either controlled by hardware (“Cache”) or by the programmer in software (“Shared Memory”)
• 16,384 32bit registers shared among 16 CUDA Cores

![[Pasted image 20251112132843.png]]


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





# 4 Memory Coarsening 

• Privatization comes with an overhead: each block needs to write to the final histogram
• Idea: Reduce the number of blocks by increasing the work of a single thread

![[Pasted image 20251119140038.png]]


Interleaving is better because it loads the data closed compact data together than contiguous partitioning


## 4.1 Coarsening with contiguous partitioning

![[Pasted image 20251119140454.png]]

## 4.2 Coarsening with interleaved partitioning: Memory Coalescing

Coarsening with interleaved partitioning for a better memory access pattern

![[Pasted image 20251119140547.png]]


# 5 Thread Cooperation: Parallel Reduction

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


## 5.1 Simple Reduction Kernel

• We start with a simple parallel reduction on the GPU
• We clearly need to cooperate among the threads to perform a tree-based reduction
• We know how to coordinate inside a block
• Let’s restrict ourselves for now to a kernel executed by a single block with B threads!
• Each thread processes two elements, therefore we can handle up to 2xB elements.

![[Pasted image 20251119140942.png]]

![[Pasted image 20251119141118.png]]


## 5.2 Minimizing control divergence

![[Pasted image 20251119141137.png]]

• Remember that threads in warps execute their instructions together
• The way we have split the work among threads in each warp, so there are quickly some threads that are active and others that are not
• Instead we can better pack the active and not-active threads together
• This reduces divergent control flow


![[Pasted image 20251119141205.png]]


## 5.3 Minimizing global memory accesses

So far we have accumulated the intermediate results in global memory, let’s use shared memory instead!

![[Pasted image 20251119141235.png]]


## 5.4 Hierarchical reduction for arbitrary input length

• So far, we only launched a single block, restricting the maximal input length
• We did this, as we can not synchronize threads across blocks
• For a reduction of arbitrary input length we need to perform a hierarchical reduction:
• We split out input into segments, each reduced independently by a block;
• then we reduce the outputs further.
• The final reduction happens on the host, in a separate kernel, or using atomics.

![[Pasted image 20251119141332.png]]

![[Pasted image 20251119141353.png]]


## 5.5 Thread coarsening for reduced overhead

• Currently, 1/2 threads are only active for loading two elements from global memory and storing their sum in shared memory
• This is very wasteful!
• Idea: increase the work for a single thread!
• This optimization is called thread coarsening (and we have seen it before in the histogram as well)

![[Pasted image 20251119141423.png]]


![[Pasted image 20251119141436.png]]


