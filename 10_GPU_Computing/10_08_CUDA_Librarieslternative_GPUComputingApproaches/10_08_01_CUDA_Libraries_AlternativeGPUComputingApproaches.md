


# 1 CUDA_Libraries
Nvidia provides a large number of libraries
These allow to utilize the GPU, often without writing CUDA kernels explicitly
CUDA organizes these into:
• CUDA Math Libraries  !
• Parallel Algorithm Libraries   !
• Data Processing Libraries
• Communication Libraries
• Quantum Libraries
• Computational Lithography
• Image and Video Libraries    !
• Deep Learning Core   !


## 1.1 CUDA Math Libraries

cuBLAS - Basic Linear Algebra
cuFFT - Fast Fourier Transform
cuRAND - Random Number Generation
cuSOLVER - Dense & Sparse Direct Solver
cuSPARSE - BLAS for sparse matrices
cuTENSOR - Tensor Linear Algebra
cuDSS - Direct Sparse Solver
CUDA Math API - Standard Math Functions
AmgX - Solver for simulations and implicit unstructured methods


![](image/Pasted%20image%2020260215220222.png)

---

CUDA Math Library - cuSPARSE

Provides basic linear algebra routines
Variation exists for structured sparsity that exploits the Tensor Cores 
Supports many different formats of sparsity
Device memory allocation using the normal CUDA malloc and memcpy API 
Additional APIs to form sparse matrices and perform operations on them

![](image/Pasted%20image%2020260215220334.png)


## 1.2 Parallel Algorithm Libraries - Thrust

Thrust is a library of parallel algorithms and data structures I
nterface very similar to C++ Standard Library
Part of a larger project to improve the CUDA experience for C++ developers, called CUDA C++ Core Libraries (CCCL)
Provides both: optimized Parallel Algorithms (reduce, scan, transform, ...) as well as Container Data Types that simplify the data management between CPU and GPU


Provided Algorithms:
- copying, gather, scatter
- merging
- prefix sum 
- reduction partition
- shuffling searching
- sorting set operations
- transformations

![](image/Pasted%20image%2020260215220747.png)

![](image/Pasted%20image%2020260215220702.png)

## 1.3 Image and Video Libraries

![](image/Pasted%20image%2020260215220805.png)


Image and Video Libraries - NPP
- NPP is a library of over 5,000 primitives for image and signal processing
Example
- Image Filters (e.g. convolution), Image Transforms,
- Geometry Functions, Statistics Functions, Signal Conversion Functions, ...

![](image/Pasted%20image%2020260215220905.png)


## 1.4 Deep Learning Core

cuDNN provides primitives to build custom neural networks by connecting operations in layers describing the network
Used (together with other libraries) by almost all ML frameworks

![](image/Pasted%20image%2020260215221952.png)

# 2 GPU Programming Approaches


So far, we used CUDA, which is the most popular GPU programming apporach
However, other approaches exist as well!


We are going look at some of them:
- Directive-based approaches: OpenACC & OpenMP
- Non-portable native approaches: NVIDIA CUDA & AMD HIP 
- Portable native approaches: OpenCL & SYCL
- Research projects: Futhark & Descend




## 2.1 Directive-Based Approaches

Idea: Annotate loops in sequential code that can run in parallel on the GPU 
Execution model: Fork-join


### 2.1.1 Introduction to OpenACC

What are compiler directives?
The directives tell the compiler or runtime to ...... 
     Generate parallel code for GPU
     Allocate GPU memory and copy input data
     Execute parallel code on GPU
     Copy output data to CPU and deallocate GPU memory


The first OpenACC directive: kernels
     ask the compiler to generate a GPU code
     let the compiler determine safe parallelism and data transfer 

![](image/Pasted%20image%2020260215225256.png)

---

The first OpenACC program: SAXPY
Example: Compute a*x + y, where x and y are vectors, and a is a scalar.
![](image/Pasted%20image%2020260215225313.png)

----
Pointer aliasing in C (1)
An improper version of the SAXPY code (using pointers):
    Pointer aliasing: Different pointers are allowed to access the same object. This may induce implicit data dependency in a loop.
    In this case, it is possible that the pointers x an y access to the same object. Potentially there is data dependency in the loop.

![](image/Pasted%20image%2020260215225343.png)

The compiler refuses to parallelize the loop that is involved with pointer aliasing. 
Compiling output of the improper SAXPY code:
Solution: Use restrict keyword to guarantee absence of aliasing ... or use the parallel directive.
![](image/Pasted%20image%2020260215225445.png)


---

Parallel directive
A proper version of SAXPY code (using parallel loop directive):
    The parallel directive tells the compiler to create a parallel region. But differently from the kernels region, the code in the parallel region (the loop in this case) is executed (by all gangs) redundantly. There is no work sharing (among gangs)
     It is necessary to add the keyword to loop share the works (among gangs). 
    In Fortran, the keyword loop can be replaced by do here.
    In C, the keyword loop can be replaced by for.

![](image/Pasted%20image%2020260215225523.png)


---

kernels vs. parallel
Parallelize a code block with two loops:
![](image/Pasted%20image%2020260215225652.png)

 Generate two kernels
 There is an implicit barrier between the two loops: the second loop will start after the first loop ends.

 Generate one kernel
 There is no barrier between the two loops: the second loop may start before the first loop ends. (This is different from OpenMP).

---


Laplace Solver (serial C)
![](image/Pasted%20image%2020260215225712.png)


![](image/Pasted%20image%2020260215225725.png)


---
Analysis of performance (version 1)
Compare the computation time (for 1000*1000 grid): 
• Serial code: 17.610445 seconds.
• OpenACC code (version 1): 48.796347 seconds

 The OpenACC code is much slower than the serial code. What went wrong? 
 We need to further analyze the parallelism and data transfer.


Analysis of data transfer (version 1)
![](image/Pasted%20image%2020260215225816.png)


----
Data clauses
copy (list): Allocates memory on GPU and copies data from host to GPU when entering region and copies data to the host when exiting region.
copyin(list): Allocates memory on GPU and copies data from host to GPU when entering region. copyout(list): Allocates memory on GPU and copies data to the host when exiting region. create(list): Allocates memory on GPU but does not copy.
present(list): Data is already present on GPU.

• Syntax for C
`#pragma acc data copy(a[0:size]) copyin(b[0:size]), copyout(c[0:size]) create(d[0:size]) present(d[0:size])`
• SyntaxforFortran
`!$acc acc data copy(a(0:size)) copyin(b(0:size]), copyout(c(0:size)) create(d(0:size)) present(d(0:size)) !$acc end data`
• If the compiler can determine the size of arrays, it is unnecessary to specify it explicitly.

![](image/Pasted%20image%2020260215225900.png)

----

Analysis of performance (version 2)
Compare the computation time (for 1000*1000 grid): • Serial code: 17.610445 seconds.
• OpenACC code (version 1): 48.796347 seconds
• OpenACC code (version 2): 2.592581 seconds
faster because we don’t copy data anyway
The OpenACC code (version 2) is around 6.8 times faster than the serial code. Cheers!
The speed-up would be even larger if the size of the problem increase.
The maximum size of GPU memory (typically 6 GB or 12 GB) is much smaller than regular CPU memory (e.g. 128 GB on BU SCC)


## 2.2 Non-Portable Native Approaches

In this class, we used the most popular programming approach CUDA 
CUDA is not-portable and native to NVIDIA GPUs
AMD has an equivalent approach called HIP (Heterogeneous-Compute Interface for Portability)
"HIP is a C++ Runtime API and Kernel Language that allows developers to create portable applications for AMD and NVIDIA GPUs from single source code."


Similar host code for memory management and kernel launches ![](image/Pasted%20image%2020260215224458.png)


Virtually identical kernel language
![](image/Pasted%20image%2020260215224514.png)

From CUDA to HIP
HIP has been build very much as a target to convert CUDA code into
The HIPIFY tool aids the automatic translation of CUDA into HIP code
AMD provides a guide* on how to convert CUDA into HIP code
They even provide a matching set of libraries
![](image/Pasted%20image%2020260215224601.png)

## 2.3 Portable Native Approaches

There exists also portable native GPU computing approaches, must notably: OpenCL and SYCL
SYCL ist the more modern standard embedded into C++
There are multiple compiler implementations for targeting different GPUs and multicore CPUs

![](image/Pasted%20image%2020260215224715.png)

---

SYCL
Memory transfers are automatically managed (only on supported GPUs)
Kernels are issued into a queue and launched asynchronously

SYCL Data Management
- Memory can also be managed more explicitly using buffers and accessors
- Buffers define how the data is laid out in memory
- Accessors describe how data is read from or written to the memory
- The SYCL runtime schedules data movement, based on the information how buffers are accessed by multiple kernels and the host
![](image/Pasted%20image%2020260215225039.png)

![](image/Pasted%20image%2020260215225119.png)


# 3 Futhark and Descend

Futhark, is a purely functional data-parallel array programming
Descend, is a safe imperative GPU programming language

## 3.1 Futhark

A purely functional data-parallel array programming language
Functional language with data-parallel pattern to express computations 

Example: Matrix Matrix Multiplication
![](image/Pasted%20image%2020260215224913.png)

Higher-level language with no explicit GPU kernels
=( less control over what computation one thread performs and which memory a thread processes

---
Optimizing Compiler
The Futhark compiler performs a number of transformations with the aim to generate fast and efficient GPU kernels
The most notable optimizations are:
Fusion to combine multiple data-parallel patterns removing the need for intermediate storage
Flattening to address the irregularity of nested data-parallelism

## 3.2 Descend

A safe imperative GPU programming language
- Imperative language with great level of control what each thread does when
- Adds memory safety guarantees
- Inspired by Rust's type system and ownership model
- Guarantees that GPU kernel are data race and deadlock free!
- Developed here at TU Berlin in our group. https://descend-lang.org/

