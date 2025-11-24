
# 1 Introduction

• CUDA Steams allows us to perform multiple CUDA operations concurrently
• A stream is a sequence of operations that executes in order on the GPU
• The operations in multiple streams may execute simultaneously

![[Pasted image 20251124105339.png]]


![[Pasted image 20251124105346.png]]


# 2 Default Stream

• By default all CUDA operations are performed in Stream 0
• Most operations are synchronous w.r.t. host and device
• Asynchronous operations w.r.t. host:
	• Kernel launches
	• `cudaMemcpy*Async / cudaMemset*Async`
	• cudaMemcpy on the same device


![[Pasted image 20251124105448.png]]

Synchronous
```
cudaMalloc ( &dev1, size ) ;
double* host1 = (double*) malloc ( &host1, size ) ;
…
cudaMemcpy ( dev1, host1, size, H2D ) ;
kernel2 <<< grid, block, 0 >>> ( …, dev2, … ) ;
kernel3 <<< grid, block, 0 >>> ( …, dev3, … ) ;
cudaMemcpy ( host4, dev4, size, D2H ) ;
...
```


Asynchronous, No Streams
```
cudaMalloc ( &dev1, size ) ;
double* host1 = (double*) malloc ( &host1, size ) ;
…
cudaMemcpy ( dev1, host1, size, H2D ) ;
kernel2 <<< grid, block >>> ( …, dev2, … ) ;
some_CPU_method ();
kernel3 <<< grid, block >>> ( …, dev3, … ) ;
cudaMemcpy ( host4, dev4, size, D2H ) ;
...
```


# 3 Multiple Streams

• Specify a stream for each CUDA operation
• cudaMemcpyAsync must use host ‘pinned’ memory, allocated with cudaMallocHost

 host ‘pinned’ memory:   please do not move the memory away

![[Pasted image 20251124105631.png]]

```
cudaStream_t stream1, stream2, stream3, stream4 ;
cudaStreamCreate ( &stream1) ;
...
cudaMalloc ( &dev1, size ) ;
cudaMallocHost ( &host1, size ) ; // pinned memory required on host

- `cudaMalloc`：在 GPU（device）上分配内存  
- `cudaMallocHost`：分配 **pinned memory（锁页内存）**，这是必须的  
    → **pinned memory 才允许真正的异步 memcpy**  
    → 如果 host 内存不是 pinned，`cudaMemcpyAsync` 会变成同步的
所以这里的 pinned memory 是 H2D/D2H 异步执行的必要条件。

…
cudaMemcpyAsync ( dev1, host1, size, H2D, stream1 ) ;
kernel2 <<< grid, block, 0, stream2 >>> ( …, dev2, … ) ;
kernel3 <<< grid, block, 0, stream3 >>> ( …, dev3, … ) ;
cudaMemcpyAsync ( host4, dev4, size, D2H, stream4 ) ;
some_CPU_method ();
...
```

Fully asynchronous / concurrent
Data used by concurrent operations should be independent

Kernel 2 and kernel 3 should be accomplished , then move the data into host 


|Stream|Operation|Overlap?|
|---|---|---|
|stream1|H2D memcpy|✔ 可与 kernel 并行|
|stream2|kernel2|✔ 可与其他 kernel 和 memcpy 并行|
|stream3|kernel3|✔ 并行|
|stream4|D2H memcpy|✔ 可与 kernel 并行|
|CPU|some_CPU_method()|✔ 不等待 GPU|

## 3.1 创建 stream 
**`cudaStream_t stream1;` —— 声明变量**
这行代码只是：
- 在 **CPU 端内存** 中声明了一个变量 `stream1`
- 类型是 `cudaStream_t`（本质是一个指向 CUDA stream 的句柄/指针类型）

👉 **这行代码不会创建 GPU stream，也不会分配 GPU 资源。**


cudaStreamCreate(&stream1); 的意思是：
👉 **向 CUDA 运行时申请创建一个新的 stream，并把创建好的 stream 放入变量 `stream1` 中。**
创建一个 CUDA 流（stream）并把句柄存到 stream1 中
- 让 CUDA Runtime **在 GPU 上创建一个新的 stream**
- 并把 stream 的句柄 **写入到 `stream1` 中**

| 代码                            | 作用                                  |
| ----------------------------- | ----------------------------------- |
| `cudaStream_t stream1;`       | 声明一个变量，用来保存 stream 的句柄（此时没有 stream） |
| `cudaStreamCreate(&stream1);` | 实际在 GPU 创建一个流，并把流的 ID/句柄放入 stream1  |

## 3.2 分配 GPU 内存、Pinned Host 内存
```
cudaMalloc(&dev1, size);
cudaMallocHost(&host1, size); // pinned memory
```

- `cudaMalloc`：在 GPU（device）上分配内存
- `cudaMallocHost`：分配 **pinned memory（锁页内存）**，这是必须的  
    → **pinned memory 才允许真正的异步 memcpy**  
    → 如果 host 内存不是 pinned，`cudaMemcpyAsync` 会变成同步的


所以这里的 pinned memory 是 H2D/D2H 异步执行的必要条件。


## 3.3 **异步 Host-to-Device 数据传输**

```
cudaMemcpyAsync(dev1, host1, size, H2D, stream1);
```

这表示：
- 从 CPU 内存（host1）复制到 GPU 内存（dev1）
- **使用 stream1**
- 异步执行，不阻塞 CPU

GPU 可以在复制同时做其他工作（比如 kernel 运行）。


## 3.4 在不同 stream 上启动多个 kernel

```
kernel2<<<grid, block, 0, stream2>>>(..., dev2, ...);
kernel3<<<grid, block, 0, stream3>>>(..., dev3, ...);

```

关键点：
- kernel2 在 **stream2** 上执行
- kernel3 在 **stream3** 上执行
- 彼此 **互不等待，可并行执行**

只要：
- GPU 有足够资源（SM）
- kernel 之间无数据依赖

这是 **并行 kernel execution**。

## 3.5 **异步 Device-to-Host 数据传输**

```
cudaMemcpyAsync(host4, dev4, size, D2H, stream4);
```

- 异步从 GPU → CPU 拷贝    
- 放在 **stream4**，所以也能与前面 kernel 并行


# 4 Explicit Synchronization

• Synchronize everything with cudaDeviceSynchronize()
• Synchronize a specific stream with cudaStreamSynchronize(streamid)
• Synchronize using Events
	• cudaEventRecord(event, streamid)
	• cudaEventSynchronize(event)
	• cudaStreamWaitEvent(stream, event)
	• cudaEventQuery(event)

Explicit Synchronization Example
Resolve using an event
```
{
	cudaEvent_t event;
	cudaEventCreate (&event); // create event
	
	cudaMemcpyAsync ( d_in, in, size, H2D, stream1 ); // 1) H2D copy of new input
	cudaEventRecord (event, stream1); // record event
	
	cudaMemcpyAsync ( out, d_out, size, D2H, stream2 ); // 2) D2H copy of previous result
	
	cudaStreamWaitEvent ( stream2, event ); // wait for event in stream1
	kernel <<< , , , stream2 >>> ( d_in, d_out ); // 3) must wait for 1 and 2
	
	asynchronousCPUmethod ( … ) // Async GPU method
}
```

cudaEventRecord (event, stream1); // record event
copy the data from  the pervious loop  in GPU to the host 


cudaStreamWaitEvent ( stream2, event ); // wait for event in stream1
Copy the data from host . Those data will be processed in the stream2  in current process. This the reason why stream2 need to wait until the event1 is done 

# 5 Understanding GPU Performance

• We want to better understand the performance of GPU programs
• For that, we are going to:
	• Discuss, performance guidelines and CUDA best practices
	• Discuss how the computer architecture allocates and schedules resources
	• Learn how to establish upper bounds of performance
	• Learn how to profile GPU programs using Nsight Compute


## 5.1 performance guidelines and CUDA best practices

![[Pasted image 20251124112206.png]]


### 5.1.1 Find Ways To Parallelize Sequential Code

• Often the hardest job!
• Not for embarrassingly parallel problems, where each iteration of a loop is independent
• There are many clever parallel algorithms for problems that are non-obvious to parallelize
• An example is the prefix-sum computation

![[Pasted image 20251124112246.png]]


### 5.1.2 Minimize Data Transfers Between Host and Device

• The bus between CPU and GPU is comparably slow: 
Tesla T4 uses a x16 PCIe Gen3 connection, with 16 GB/s The DDR6 GPU global device memory has 320 GB/s
• Avoid moving the same data multiple times, e.g., during iterative processes
• Overlap data transfer and computation to hide the transfer cost
=>" use CUDA streams for that

### 5.1.3 Maximize Device Utlization: adjust kernel launch configuration to maximized device utilization


### 5.1.4 Ensure Global Memory Accesses Are Coalesced

Reminder from prior Lecture
• When a warp executes an instruction that accesses global memory, the individual memory accesses are coalesced into one or more memory transactions
• Memory transactions are either 32-, 64-, or 128-byte wide
• It is particularly beneficial when the threads in the warp access consecutive memory locations, as the least number of memory transfers must be performed

![[Pasted image 20251124112519.png]]


### 5.1.5 Minimize redundant accesses to global memory whenever possible 

use registers or shared memory to hold frequently accessed data 

read the data from neighbooring memory 
read memory using coalescing/ share the memory together 

### 5.1.6 Avoid Warp Divergence within in the same warp 

Reminder: Execution Model
• The warp scheduler finds the threads from a warp that perform the same instruction and schedule them together:
• When execution diverges some threads pause execution

![[Pasted image 20251124113120.png]]

---

Reminder from prior Lecture
• Remember that threads in warps execute their instructions together
• The way we have split the work among threads in each warp, so there are quickly some threads that are active and others that are not
• Instead we can better pack the active and not-active threads together
• This reduces divergent control flow

![[Pasted image 20251124113214.png]]


# 6 Device Utilization & Occupancy

• We know that all threads within a block are guaranteed to execute on one SM
• Therefore, these threads must share the resources of the SM, in fact multiple block might execute on the same SM
• Ideally, we have a lot of warps ready to execute to hide memory latency
• We call the ratio of the number of warps assigned to an SM to the maximum number the SM supports the occupancy.   how much ressource is in use 
• Why would we ever have an occupancy of less than 100%?


## 6.1 Shared Resources of an SM   Single multiproccessor
 
• One SM can:
	• manage up to 16 blocks (each block has up to 1024 threads)
	• manage up maximum to 32 warps = 1024 threads, whichever how many blocks are used 
• One SM has:
	• 64k 个 32-bit registers
	• 32 or 64KB shared memory (can be configured)
• We must consider all these limits when choosing our launch configuration.

Per SM:
max 16 blocks (each block has up to 1024 threads)
max 32 warps
max 1024 threads

---

Bad Occupancy due to Bad Launch Configs

• Bad Example 1:
We launch a grid with blocks of 32 threads each => " SM executes 16 * 32 = 512 threads. 
Occupancy: 50% (= 512 executed threads / 1024 possible threads)

• Bad Example 2:
Blocks of size 768 threads =" SM executes 1 block `(as 2 block = 2*768>1024 threads) `
`Occupancy: 75% (= 768 / 1024)`


---
Bad Occupancy due to Memory Resource Constraints

• To run at full occupancy, each thread can not require more than 65,536 （一共有多少个寄存器） / 1024 （最多有多少个 thread）= 64 registers.
• All blocks on the SM share the same shared memory

Bad Example 3:
• Blocks of size 512, each block allocates 35KB shared memory
=" SM executes 1 block (as 2*35KB > 64 KB shared memory)
Occupancy 50% (= 512 / 1024)



# 7 Nsight Compute


• Nsight Compute is Nvidia’s profiler for CUDA code
• It can greatly help to help understand the performance of GPU applications and identify bottlenecks

![[Pasted image 20251124113528.png]]

![[Pasted image 20251124113537.png]]

![[Pasted image 20251124113652.png]]

## 7.1 Profiling With Nsight Compute

• Nsight Computer is a GUI application to look at the recorded profiles

• Profiles can be recorded using the ncu command line tool:
```
sudo /usr/local/cuda/bin/ncu -o profile -&set full ./CUDAapplication
```

• The tool needs root rights to perform the profiling
(more details here: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

• Profiling significantly slows down the overall application, but doesn’t impact the runtime of the kernel very much.


Investigating the Recorded Profile With the GUI
![[Pasted image 20251124113757.png]]