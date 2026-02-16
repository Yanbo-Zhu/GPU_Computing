
# 1 Computer Microarchitecture

The ISA (Instruction Set Architecture) describes the interface between software and hardware
• The Microarchitecture implements the ISA in form of digital logic
• It consists of registers, memory, arithmetic logic units, and other digital logic blocks.
• The microarchitecture describes features not exposed to the software, such as:

• Pipelining
• Out-of-order execution
• Speculative execution
• Caching
• Memory access policies
• Prefetching
• Voltage scaling
• Error correction

# 2 CPU vs GPU Design


• CPU and GPU both follow a common computer architecture
• But, they decide to use the available transistors very differently
• They follow two different architecture design philosophies

![[Pasted image 20251107135757.png]]


# 3 Fundamental Challenge of Computer Architecture (Latency and limited throughput  when data transferring)


Von Neumann Architecture
• Fundamental idea of (almost) all modern computer
architecture:
• Store data and instructions together in memory
• Execution pipeline:  Fetch Instruction ->- Decode ->- Fetch Operands ->- Execute ->- Write Result

• Problem “von Neumann Bottleneck”: Latency and limited throughput when transferring between memory and processor units.
==• Result: The processing units often are forced to wait (“stall”)==

![[Pasted image 20251107140342.png]]

Possible solutions to improve performance of a single ALU:
• Speculative execution via branch prediction
• Prefetch (load in advance) memory that is likely to be used soon
• Cache 

All these solutions aim to reduce the possibility of stalls and time a single ALU has to wait for memory


# 4 Latency Oriented Processor Architecture

This solutions aimed at executing a single serial thread of instructions as fast as **possible**
AIM : Execute a single serial thread of instructions as fast as possible


Intel Core i9-13900K
![[Pasted image 20251107140508.png]]

![[Pasted image 20251107141754.png]]



## 4.1 Parallelism in Processor Architecture

This solutions aimed at executing a single serial thread of instructions as fast as possible


There are multiple levels at which a processor can exploit parallelism
1. Bit-level parallelism
2. Pipelining parallelsim
3. Instruction-level parallelism
4. Task parallelism
5. Data parallelism
• All levels are exploited in modern CPUs, however levels 4 & 5 requires software change



### 4.1.1 Bit-Level Parallelism
• We group bits to bytes, ie., groups of 8bits
• In a processor we further group multiple bytes to a word-size of 64bits
• This is the unit at which most modern processors operate instructions
• It is not possible to perform an instruction only on a single bit, the processor will always load the entire word into a register, operate on it, and write it back
• We perform, e.g., 64bit floating point operations in a single clock cycle



### 4.1.2 Pipelining Parallelism

• Idea comes from the assembly line:
multiple people assemble a series of cars at the same time,
each performing a different task
• In a modern super-pipelined processor, executing of an instruction is
broken down into 15-20 sequential steps, simplified

Fetch Instruction ->- Decode ->- Fetch Operands ->- Execute ->- Write Resul

![[Pasted image 20251107140917.png]]


![[Pasted image 20251107140947.png]]


### 4.1.3 Instruction-Level Parallelism

• Originally, processors had only a single functional unit (FU) to perform
computations, such as an ALU (Arithmetic logical unit) or a FPU (floating
point unit). These processors were called scalar processors.
• Almost all processors now have multiple function units within a single core.
• When the hardware simultaneously dispatches multiple instructions to
different execution units, we call this a superscalar processor

![[Pasted image 20251107141050.png]]

- There are alternatives to detecting dependencies in the hardware at runtime
- VLIW (Very Long Instruction Word) and EPIC (Explicitly parallel instruction computing): instead of detecting the dependencies between instructions in hardware, the software or compiler must declare/detect dependencies ahead of time
	- Advantage: simpler hardware
	- Disadvantage: more complex software and compilers

Today no mainstream VLIW or EPIC architectures exist


### 4.1.4 Task Parallelism


• All prior techniques work for performing a single task
• If the software performs multiple task and is written in a way that exposes multiple threads, or we execute multiple processes, we can leverage:
• Multi-core processors (MCP): multiple cores perform independent executions of threads. Each core has it’s own execution pipeline.
• Cores can either all be the same (homogeneous), or different (heterogeneous)
• Simultaneous multithreading (SMT): instructions from more than one thread can be executed in any given pipeline stage at a time.


### 4.1.5 Data parallelism

• Perform a single task working on multiple data item simultaneously
• One technique to exploit data parallelism are SIMD vector instructions
- 同时在多个数据项上执行同一个任务。    
- 一种利用数据并行性的技术是 **SIMD 向量指令**。

• Flynn's taxonomy:
	• SIMD - Single instruction, multiple data
	• SISD - Single instruction stream, single data stream
	• MIMD - Multiple instruction streams, multiple data streams
	• MISD - Multiple instruction streams, single data stream



SIMD vector instructions
• Vector instructions are specific instructions to perform floating point operations on a fixed-size collection of multiple floating point values
• In modern CPUs up to 16 32bit float values are grouped into a vector
• A single instruction then performs, e.g., a multiplication of two such vectors
• These instructions are executed by a hardware unit in a single clock cycle
• Requires either the software to use these instructions directly, or the compiler to automatically vectorize the scalar code by transforming it using these instructions

SIMD 向量指令
- 向量指令是一种专门用于对**固定大小的多个浮点值集合**执行浮点运算的指令。
- 在现代 CPU 中，最多可将 **16 个 32 位浮点数** 组合成一个向量。
- 一条指令可以同时对两个这样的向量执行例如乘法的操作。
- 这些指令由硬件单元在**一个时钟周期**内完成执行。
- 这要求软件要么**直接使用这些指令**，要么由**编译器自动向量化**标量代码（即通过转换代码来利用这些指令）。


# 5 Thoughput Oriented Processor Architecture

Alternative approach on how to keep the arithmetic logical units busy?
Prior solutions aimed at executing a single serial thread of instructions as fast as possible

• What if we instead focus on executing many threads while attempting to maximize the overall throughput, even though sacrificing the performance of each single serial thread may be required.

Aim: Maximize overall throughput of instructions, possibly at the cost of the performance of an individual thread.
• For this to work, there must be ample parallelism exposed to the hardware!
• Examples: GPUs, ML-accelerators

## 5.1 Nvidia Tesla T4 GPU

![[Pasted image 20251107143023.png]]


![[Pasted image 20251107143031.png]]


![[Pasted image 20251107143056.png]]

![[Pasted image 20251107143106.png]]

## 5.2 How to keep the arithmetic logical units busy?

• How do we prevent ALUs to stall when waiting for a memory operation in a throughput oriented architecture?
	• Idea: we don’t, we just do something else while we are waiting!
	• Exploit parallelism to hide the memory latency


• Interleave execution of multiple threads
• Keep state of threads in registers to make switching between registers cheap!
• That’s why the Tesla T4 GPU has over 18 MB of Registers on the chip instead of a larger Cache!


![[Pasted image 20251107143204.png]]


stall （使）熄火，抛锚；故意拖延（以赢得时间）；拖住（某人）；暂缓，搁置，停顿；（飞机）失速；（航海）（由于风力不够而）航行失控；把（牲畜）关在厩内（养肥）


# 6 CUDA

## 6.1 CUDA Cores 

sm: single multiprocesser 
warp: This group of 32 threads

• 64 CUDA Cores per SM
• One CUDA Core can execute a INT32 and FP32 instruction simultaneously
• Each CUDA Core executes its own thread of execution
• For execution 32 threads are grouped and 32 CUDA Cores perform the same instruction in a clock cycle
• This group of 32 threads is called a warp


![[Pasted image 20251107143549.png]]


## 6.2 Execution Model
• The warp scheduler finds the threads from a warp that perform the same instruction and schedule them together:
• When execution diverges some threads pause execution

![[Pasted image 20251107143759.png]]


## 6.3 SIMT - Single Instruction, Multiple Threads

SIMT - Single Instruction, Multiple Threads  
• What is the difference between SIMD and SIMT?  
• In SIMD, a single instruction acts upon all the data in exactly the same way.  
• In SIMT, selected threads can be activated or deactivated, allowing more   ﬂexible executions including branching.  
• Highest efﬁciency is achieved when all threads perform the same instruction  
• Avoiding these divergences is going to be a goal when writing GPU software

- SIMD 和 SIMT 的区别是什么
    - 在 **SIMD** 中，一条指令以**完全相同的方式**作用于所有数据。
    - 在 **SIMT** 中，可以**选择性地激活或停用线程**，从而实现更灵活的执行方式，包括**分支（branching）**操作。
- 当所有线程执行**相同指令**时，可以达到**最高效率**。
- 在编写 GPU 程序时，**尽量避免线程间的分歧（divergence）** 是一个重要目标。


## 6.4 Tensor Cores


• Each SM has 8 Tensor Cores
• Each Tensor Core can perform in a single clock cycle:
	• a 4x4 FP16 Matrix Multiplication
	• a 4x8 INT8 Matrix Multiplication
	• a 4x16 INT4 Matrix Multiplication
• These are crucial for Machine Learning

## 6.5 Memory of Nvidia Tesla T4 GPU

• The entire GPU has access to 16 GB of memory
• L2 Cache of 6MB shared among all 72 SMs / 4,608 CUDA Cores
• L1 Cache / Shared Memory of 96KB shared among 64 CUDA Cores in one SM
	• Can be configured to be either controlled by hardware (“Cache”) or by the programmer in software (“Shared Memory”)
• 16,384 32bit registers shared among 16 CUDA Cores


![[Pasted image 20251107144457.png]]



# 7 Nvidia GPU Microarchitectures

![[Pasted image 20251107144251.png]]


## 7.1 Nvidia H100 GPU

![[Pasted image 20251107144529.png]]

![[Pasted image 20251107144536.png]]


![[Pasted image 20251107144549.png]]


# 8 How Do We Keep All These Cores and Hardware Units Busy?


• GPUs assume that there is enough data parallelism to keep all the cores busy
• It is the programmers job to expose the parallelism to the GPU!
• We must launch a significant number of threads to have work for each core and hide memory latency!
• Launching the right number of threads is an optimization challenge, we are going to discuss later in this course

- GPU 假设任务中具有**足够的数据并行性**，以便让所有核心都保持忙碌。
- 程序员的职责是**将并行性暴露给 GPU**！
- 我们必须**启动足够多的线程**，以确保每个核心都有任务可做，同时**隐藏内存访问延迟**。
- 启动**合适数量的线程**是一项**优化挑战**，我们将在本课程后面进一步讨论这一点。




