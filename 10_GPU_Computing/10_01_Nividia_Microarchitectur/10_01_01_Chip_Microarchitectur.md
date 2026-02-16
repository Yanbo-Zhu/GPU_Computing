

# 1 Nvidia Curie Microarchitecture (2004)

![[Pasted image 20251107134341.png]]


## 1.1 The graphics pipeline

![[Pasted image 20251107134827.png]]


![[Pasted image 20251107134842.png]]


1 Vertex processing

Vertices are transformed into “screen space”

![[Pasted image 20251107134554.png]]


2 Primitive processing/ Primitive Assembly

Then organized into primitives that are clipped and culled…
primitive 图元 

![[Pasted image 20251107134618.png]]


3 Rasterization
栅格化，光栅化

Primitives are rasterized into “pixel fragments”

![[Pasted image 20251107134720.png]]


4 Fragment processing

![[Pasted image 20251107134755.png]]

5  Pixel operations

Fragments are blended into the frame buffer at their pixel locations (z-buffer determines visibility)

![[Pasted image 20251107135123.png]]

# 2 Nvidia Tesla Microarchitecture (2006)  (unified shader architecture)

shader: 著色器；着色程序

n 2006, Nvidia releases the first “unified shader architecture” and a programming language CUDA (originally Compute Unified Device Architecture) allowing programmers to perform non-graphics computations

![[Pasted image 20251107135256.png]]

![[Pasted image 20251107135316.png]]


![[Pasted image 20251107135324.png]]



