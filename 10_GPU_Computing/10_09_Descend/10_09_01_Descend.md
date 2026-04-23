
A Safe GPU Systems Programming Language
"if the program successfully compiles,  then it is free of data races and other memory problems"
# 1 

![](image/Pasted%20image%2020260216202718.png)

![](image/Pasted%20image%2020260216202742.png)


GPU kernel functions are executed by thousands of threads
GPU kernel functions are executed by thousands of threads
Threads are hierarchically organized into multi-dimensional blocks   Blocks are hierarchically organized into a multi-dimensional grid


## 1.1 Is this access safe?

![](image/Pasted%20image%2020260216202833.png)

![](image/Pasted%20image%2020260216202908.png)


# 2 Descend


"if the program successfully compiles,  then it is free of data races and other memory problems"

## 2.1 References are unique or shared and carry the address space of the multi-dimensional array as part of their type
![](image/Pasted%20image%2020260216203154.png)


---

## 2.2 Functions are annotated with the execution resource that will execute the function
![](image/Pasted%20image%2020260216203621.png)

---

## 2.3 
Collectively executed by the entire grid

![](image/Pasted%20image%2020260216203940.png)

Collectively executed by each block in the grid
![](image/Pasted%20image%2020260216203956.png)

Collectively executed by each thread in each block
![](image/Pasted%20image%2020260216204023.png)


---
## 2.4 Data-race free parallel memory accesses using views

![](image/Pasted%20image%2020260216204308.png)


![](image/Pasted%20image%2020260216204354.png)

![](image/Pasted%20image%2020260216204402.png)


![](image/Pasted%20image%2020260216204413.png)

![](image/Pasted%20image%2020260216204422.png)

![](image/Pasted%20image%2020260216204526.png)

![](image/Pasted%20image%2020260216204536.png)

## 2.5 Descend has five basic view primitives

![](image/Pasted%20image%2020260216205055.png)



## 2.6 Views are composed to express complex memory access patterns


![](image/Pasted%20image%2020260216205132.png)

## 2.7 Views to express more complex memory access patterns

![](image/Pasted%20image%2020260216205200.png)

## 2.8 Descend ensures the presence of all required synchronization

![](image/Pasted%20image%2020260216205322.png)

![](image/Pasted%20image%2020260216205328.png)


# 3 How does Descend  achieve this safety?

Descend's type system with extended borrow checking
- Execution Resources： formally represent  groupings of  blocks and threads
- Place Expressions are unique names  for memory objects
- Memory Views are safe parallel  access patterns

## 3.1 Execution Resources

Instructions are executed in various contexts on the GPU, e.g.,
- a memory access might be out of bound depending on the number of threads in a block
- a barrier synchronization must be performed by all threads in a block

Descend's type system tracks who owns values and which instructions are executed by the grid, blocks, warps, or threads.

 For this, we need to be able to syntactically compare execution resources.

指令在 GPU 上会在各种不同的上下文中执行，例如：
根据线程块中的线程数量，某次内存访问可能会越界。
屏障同步必须由线程块中的所有线程共同执行。

Descend 的类型系统会追踪"谁拥有这些值"以及"哪些指令是由网格、线程块、线程束或线程执行的"。
为此，我们需要能够在语法层面比较执行资源。

![](image/Pasted%20image%2020260216210327.png)

## 3.2 Place Expressions and Views

Introduced by Rust as a unique name for a memory object.

Aliases are resolved by substituting the referenced place expressions.

This allows them to be compared syntactically in to ensure that the same memory location is not (mutably) accessed simultaneously.

The result of applying a view to an array is a view-array.

View-arrays are very similar to ordinary arrays, but they are not guaranteed to be contiguous in memory.
 Parallel accesses via views are guaranteed to be safe, i.e., data race free.

![](image/Pasted%20image%2020260216210357.png)

由 Rust 引入，作为内存对象的唯一名称（Unique Name）。

别名（Aliases）通过替换所引用的位置表达式来解析。

这使得它们能够在语法层面进行比较，以确保同一内存位置不会被同时（可变地）访问。

将视图（View）应用于数组得到的结果是一个视图数组（View-Array）。

视图数组与普通数组非常相似，但不保证在内存中是连续存储的。

通过视图进行的并行访问是安全的，即不会发生数据竞争。

## 3.3 Extended borrow checking

In Rust the borrow checker checks if a thread is allowed to create a (unique or shared) reference to a memory object, i.e., "borrow" it.

 In Descend, each execution resource can take ownership of a memory object or might borrow.


For a single thread to have exclusive memory access, the ownership and borrows must be narrowed.

Narrowing describes how ownership and borrows are refined when navigating the execution hierarchy from grid to threads.

在 Rust 中，借用检查器（Borrow Checker）用于检查某个线程是否允许创建对某个内存对象的（唯一或共享）引用，也就是"借用"它。

而在 Descend 中，每个执行资源（Execution Resource）都可以拥有某个内存对象的所有权，或者进行借用。

为了使单个线程拥有独占的内存访问权，所有权和借用必须被收窄（Narrowed）。

"收窄"描述了：当从网格（Grid）层次向下深入到线程（Threads） 的执行层次结构时，所有权和借用关系是如何被细化（Refined） 的。


Narrowing can be violated:
![](image/Pasted%20image%2020260216210703.png)

Descend's extended borrow checker performs narrowing checks to ensure safety.

# 4 Descend's type system

![](image/Pasted%20image%2020260216210744.png)

![](image/Pasted%20image%2020260216210751.png)

# 5 Does it allow to write fast programs?


![](image/Pasted%20image%2020260216205938.png)

![](image/Pasted%20image%2020260216205948.png)

![](image/Pasted%20image%2020260216205955.png)
# 6 What's next?

Soundness proof of the type system, including a proof of deadlock freedom for well-typed programs
Improve our implementation and implement more applications 
Model more advanced features of the GPU, e.g., Tensor Cores (already done), Asynchronous Behavior, Weak Memory Model ... 

How to design a family of similar languages and type systems for other GPUs and accelerators
How do we compose these languages for programming heterogeneous systems?
How do we reduce the effort of designing and implementing them?

