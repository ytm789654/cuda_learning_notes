为了方便讨论，假设矩阵是方形的，即M=N  
将一个m x m的矩阵转置并不复杂，用一维数组存储矩阵
```C++
void matrix_trans(const float * a, float * c, int m)
{
    for(int i = 0; i<m; i++)
        for(int j = 0; j<m; j++)
            c[i * m + j] = a[j * m + i];
}
```

### step1 naive
不过既然是在学习CUDA自然要并发的处理问题，很轻易的可以想到，用一个线程处理一个元素，就能写出简单的CUDA转置:
```C++
__global__ void matrix_trans(const float * a, float * c, int m)
{
    int myRow = blockDim.y * blockIdx.y + threadIdx.y;
    int myCol = blockDim.x * blockIdx.x + threadIdx.x;
    c[myRow * m + myCol] = a[myCol * m + myRow];
}
```
使用NCU分析内存读写
```
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum

----------------------------------------------- ----------- ------------
Metric Name                                     Metric Unit Metric Value
----------------------------------------------- ----------- ------------
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                  524,288
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                  524,288
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector   16,777,216
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum       sector    4,194,304
----------------------------------------------- ----------- ------------
```
可以看到读写内存的请求次数相同但是读取的transaction远高于写。在说原因之前先贴一段官方文档
Global memory resides in device memory and device memory is accessed via 32-, 64-, or 128-byte
memory transactions. These memory transactions must be naturally aligned: Only the 32-, 64-, or 128-
byte segments of device memory that are aligned to their size (i.e., whose first address is a multiple
of their size) can be read or written by memory transactions.  
When a warp executes an instruction that accesses global memory, it coalesces the memory accesses
of the threads within the warp into one or more of these memory transactions depending on the size
of the word accessed by each thread and the distribution of the memory addresses across the threads.
In general, the more transactions are necessary, the more unused words are transferred in addition to
the words accessed by the threads, reducing the instruction throughput accordingly.  
大意是说global memory是按照32字节的倍数一次transaction读取的。warp内的线程如果读取的可以的话会合并访存的指令，大家一起用这次transaction取回的数据。

分析代码可以看到，写入c数组 c[myRow * m + myCol] 表达式中，在一个warp内，myRow由threadIdx.y决定，myCol由threadIdx.x决定，而warp内32个线程的ID连续，threadIdx.x一般连续而threadIdx.y相同，所以warp内myRow相同，而myCol递增。于是32个线程写入的内存实际上是连续的（相当于矩阵的一行）。  
相反的，读取a数组的数据，warp内myRow相同，而myCol不同，此时myCol * m + myRow的计算结果会相当跳跃，读取的内存地址并不连续，所以需要多次transaction。  

### step2 访存合并，优化读写
通过之前的分析可以看到，写入内存合并访存已经实现，但是读取的情况并不乐观。但是这两个过程由于是转置所以col和row是一定会反过来的，一面合并另一面一定无法合并，除非能找到一个办法先合并读取过程把结果存下来，再从存下的结果取出数据合并写入。要暂存数据，肯定就用到了共享内存，shared memory。  
先分析block的任务，借助共享内存，block可以处理原矩阵一定区域的数据。先将这些数据读入共享内存中，然后再将存下的数据写入到输出矩阵的对应位置即可。也就是说，block操作的对象，是一整个子矩阵。  
再分析thread的任务，为了访存连续性，所以要先读取子矩阵内row col的数据（注意是子矩阵的row col，而非原矩阵，所以可以用threadIdx标记），存入共享内存。然后为了写入访存连续性，仍然在目标子矩阵的row col处写入数据，注意，此时写入的数据应该来自于其他线程读入共享内存，所以需要block内的线程同步。那么可以写出代码
```C++
#define blockSizeX 32
#define blockSizeY 32
__global__ void matrix_trans(const float * a, float * c, int m)
{
    int tileX = blockIdx.x * blockDim.x;
    int tileY = blockIdx.y * blockDim.y;    //相当于子矩阵的起始位置
    int row = threadIdx.y;
    int col = threadIdx.x;      //子矩阵内的row和col
    __shared__ float subMatrix[blockSizeX][blockSizeY];
    subMatrix[row][col] = a[(tileY + row) * m + (tileX + col)];
    __syncthreads();
    //转置后，子矩阵的起始位置应该由 (tileY, tileX) 变为 (tileX, tileY), 同时，线程仍然对row行，按列进行连续填充子矩阵，保证写入的连续性。线程操作的子矩阵的元素相对位置不变，仍然是row行col列
    c[ (tileX + row) * m + (tileY + col) ] = subMatrix[col][row];
}
```
使用NCU查看
```
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed

结果
-------------------------------------------------------- ----------- ------------
Metric Name                                              Metric Unit Metric Value
-------------------------------------------------------- ----------- ------------
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum               15,784,589
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                  233,574
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                   16,833,165
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                    1,282,150
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                           524,288
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                           524,288
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector    4,194,304
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector    4,194,304
smsp__cycles_active.avg.pct_of_peak_sustained_elapsed              %        89.33
-------------------------------------------------------- ----------- ------------
```
可以看到l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum这一项已经降到了和写入一样的水平。但是l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum， l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum两项很高。这反应了共享内存读取有大量的bank conflict问题。

### step3 优化bank conflict
bank conflict是同一个warp内的不同线程访问同一个bank内的地址导致的，一个bank是四个字节，共有32个bank。
执行
```
c[ (tileX + row) * m + (tileY + col) ] = subMatrix[col][row];
```
时由于warp内row相同col不同，相当于在按列访问二维数组，而数组的宽度为32，于是subMatrix数组每列同一元素地址差为32 * sizeof(double) 刚好是32 * 4 的倍数这意味着每个线程都在访问同一个bank，bank conflict就发生了。  
优化其实很简单，宽度加一个padding这样按列访问的地址差是33 * sizeof(double) 不再总是32 * 4的倍数了。
```
#define blockSizeX 32
#define blockSizeY 32
__global__ void matrix_trans(const float * a, float * c, int m)
{
    int tileX = blockIdx.x * blockDim.x;
    int tileY = blockIdx.y * blockDim.y;    //相当于子矩阵的起始位置
    int row = threadIdx.y;
    int col = threadIdx.x;      //子矩阵内的row和col
    __shared__ float subMatrix[blockSizeX][blockSizeY + 1];
    subMatrix[row][col] = a[(tileY + row) * m + (tileX + col)];
    __syncthreads();
    //转置后，子矩阵的起始位置应该由 (tileY, tileX) 变为 (tileX, tileY), 同时，线程仍然对row行，按列进行连续填充子矩阵，保证写入的连续性。线程操作的子矩阵的元素相对位置不变，仍然是row行col列
    c[ (tileX + row) * m + (tileY + col) ] = subMatrix[col][row];
}
```
profiling结果为
```
-------------------------------------------------------- ----------- ------------
Metric Name                                              Metric Unit Metric Value
-------------------------------------------------------- ----------- ------------
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                  173,856
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                    1,048,576
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                    1,222,432
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                           524,288
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                           524,288
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector    4,194,304
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector    4,194,304
smsp__cycles_active.avg.pct_of_peak_sustained_elapsed              %        87.21
-------------------------------------------------------- ----------- ------------
```
可以看到起到了很好的优化效果