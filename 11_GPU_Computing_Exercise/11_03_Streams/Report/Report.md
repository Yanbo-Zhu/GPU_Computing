
# 1 command to generate report

```bash
ncu --set full --target-processes all -o my_report ./your_program
```

using matrixMultiplyWithStreams() with 
```
matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
```



# 2 Kernel report with Konfiguration 1

```bash
ncu --set full --target-processes all -o my_report_Konfiguration1  ./task
```

int MATRIX_SIZE = 4096;
int TILE_SIZE = 32;
dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);  // 128 x 128 
dim3 blocksPerGrid(MATRIX_SIZE/TILE_SIZE, MATRIX_SIZE/TILE_SIZE); // 1024 x 1024 



