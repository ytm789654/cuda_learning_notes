### reduce(规约)算法
其实也不能叫算法，更像是一种思想，一种解决问题的方式。  
举个简单的例子，数组求和。  
最普通的方法
```C++
float sum(vector<float> nums)
{
    float sum = 0.0f
    for(int i = 0; i<nums.size(); i++)
        sum += nums[i];
    return sum;
}
```
这是再寻常不过的串行计算方式，依次累加。  
但是CUDA更讲究并行的方式解决问题，把大的问题分解成并行的子问题，最理想的情况是子问题的解本身就是最终答案的一部分，比如非常经典的向量相加，CUDA可以用一个线程负责向量中对应位置元素的求和并将结果写入返回数组中。而对数组求和略有不同，很自然的可以想到将数组分段，每个线程对一段数据求和，这样才有充分利用CUDA并行计算特性的可能，但是很显然，分段的求和结果与数组求和的结果还差了一步，要将所有线程求出来的和累加起来，而这个过程就是reduce。实际上把数组分段的过程叫map，这俩凑一块就是有名的map-reduce。  
那么问题来了，怎么把每个线程计算的结果，reduce起来？  
不妨先想象kernel函数返回什么。  
CPU计算很多个数的和可能比不上GPU，但是算几百上千个数还是不成问题的，我们可以申请一块连续的空间，存放分段的结果。
比如我们使用一个数组float out[256]来存放结果的话，我们如何既大量线程并发，又最后把结果reduce起来呢？  

### step1 thread reduce
先从thread的角度思考一下，我们可以让一个thread负责计算out里的一个值，也就是把输入数据分成256段，每个thread处理一段，这很简单，只需要每个thread去分段读数据最后加起来就行了
```C++
__global__ void reduce(float *gdata, float *out, long long N){
     int tid = threadIdx.x;
     int idx = tid;
     float threadSum = 0.0;
     int stride = BLOCK_SIZE;
     while (idx < N) {  // grid stride loop to load data
        threadSum += gdata[idx];
        idx += stride;
        }
     out[tid] = threadSum;
  }
```
调用
```C++
blockNum = 1;
blockSize = 256;
reduce<<<blockNum, blockSize>>>(gdata, out, N);
```
不得不说这是个简单的方法非常的省脑细胞，但是我们可以脑测一下，如果N非常大，区区两百多个线程有点小马拉大车了，一方面访存线程过少，另一方面访存次数又过多计算也多。当然，我们可以把BLOCK_SIZE加大，不过再大也不能超过1024，对于大量的数据仍然是杯水车薪。不能充分利用好并行优势,得想办法多加点thread

### step2 bolck求和
BLOCK_SIZE确实有上限，但是我们可以加BLOCK_NUM啊。因为新加一层，我们不得不在原来的模型上多分析一层，每个BLOCK要做什么呢？
我们仍然是要处理那么多数据，填满out[256],不妨就开256个block（仍然是每个block256个线程），让每个block算一个值出来。  
问题又来了，怎么控制block里的线程协作算出一个值来呢？  
对照一下外层逻辑：host调用256个block算出的值存在out里，这是因为每个block都可以访问out。  
block里用256个线程也可以算出256个值，存在一个每个线程都能访问的地方就行了————答案呼之欲出，共享内存。
```C++
__global__ void reduce(float *gdata, float *out, long long N){
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;
    float threadSum = 0.0;
    int stride = BLOCK_SIZE * blockDim.x;
    while (idx < N) {  // grid stride loop to load data
       threadSum += gdata[idx];
       idx += stride;
       }
    sdata[tid] = threadSum;
    __syncthreads();
    // TODO: block内求和
    // out[blockIdx.x] = ???;
}
```
问题又来了，如何在block内求和？似乎很简单，让一个线程去求和就行了
```C++
    if(tid == 0){
        for(int i = 0; i<BLOCK_SIZE; i++)
            blockSum += sdata[i];
    }
```
但是要注意到，此时整个BLOCK都在SM上，一整个BLOCK内其他线程都干看着这一个线程干活，颇有一核有难七核围观的壮烈感。  
所以要找一个办法尽量提高其他线程的参与。  
求和的基本是相加，需要两个数参与，不可能让每个线程读一个数，那就只能让一个线程读两个数，结果加起来存下来，这样待求和数的规模会不停的缩小。虽然随着待求和项的减少无法充分调动block内所有线程，但是比一个线程做了所有的工作要强不少。  
于是我们每次就像把一张纸对折一样，把数组后一半加到前一半去，直到最后所有的数都加到sdata[0]上。也就是  
第一轮 前128个元素与后128个元素相加，结果存在前128个元素上  
第二轮 在第一轮所有的数据都求和到前128个元素的基础上。把其中的前64个元素与后64个元素相加  
...
第八轮 把上一轮运算的两个数据加起来，写入到s[0]得到sdata内所有元素的和。
把这个过程转换成代码，就是
```C++
for(offset = BLOCK_SIZE/2; offset>0; offset >>= 1){
    if(tid < offset)
        sdata[tid] += sdata[tid + offset];
    __syncthreads();    //这个同步非常重要，只有这样才能数据更新完成
}
```
于是所有的代码就是
```C++
__global__ void reduce(float *gdata, float *out, long long N){
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;
    float threadSum = 0.0;
    int stride = BLOCK_SIZE * blockDim.x;
    while (idx < N) {  // grid stride loop to load data
       threadSum += gdata[idx];
       idx += stride;
    }
    sdata[tid] = threadSum;
    __syncthreads();
    float blockSum = 0.0f;
    for(offset = BLOCK_SIZE/2; offset>0; offset >>= 1){
        if(tid < offset)
        sdata[tid] += sdata[tid + offset];
        __syncthreads();    //这个同步非常重要，只有这样才能数据更新完成
    }
    if(tid == 0)
        out[blockIdx.x] = s[0];
}
```

### step3 借助warp进一步优化
首先需要了解一下warp的概念。
NV的官方文档Hardware Implementation这一章的SIMT Architecture一节提到，warp由threadId相近的32个线程组成，是SM上装载block的调度单位。每个warp同时只能执行一条指令，所以如果一个warp内的thread行为出现分歧（比如类似于if(threadIdx % 2 == 0)一类的判断）那么warp只能分别执行对应的指令并让不符合判断的线程挂起(官方称这个情况为warp diverged)。这也是为什么CUDA尽量少写些判断为好的原因之一。  
warp统一行动的一个好处，就是warp内部的线程不用同步————毕竟大家都在尽量执行同一条指令  
基于这个特性，可以对reduce进行进一步的优化
```C++
__global__ void reduce(float *gdata, float *out, long long N){
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;
    float threadSum = 0.0;
    int stride = BLOCK_SIZE * blockDim.x;
    while (idx < N) {  // grid stride loop to load data
       threadSum += gdata[idx];
       idx += stride;
    }
    sdata[tid] = threadSum;
    __syncthreads();
    float blockSum = 0.0f;
    for(int offset = BLOCK_SIZE/2; offset>32; offset >>= 1){
        if(tid < offset)
        sdata[tid] += sdata[tid + offset];
        __syncthreads();    //这个同步非常重要，只有这样才能数据更新完成
    }
    if(tid < 32){   //这个判断不会引起warp diverged 因为符合情况的thread恰好在一个warp中。从这里开始请从warp角度思考一下代码怎么写
        warp[tid] += warp[tid + 16];
        warp[tid] += warp[tid + 8];
        warp[tid] += warp[tid + 4];
        warp[tid] += warp[tid + 2];
        warp[tid] += warp[tid + 1];
        if(tid == 0)
            out[blockIdx.x] = s[0];
    }
}
```
不难发现其实只是展开了一部分循环，但是减少了__syncthreads()调用因为不需要等待其他线程，会走到这个判断里的线程行为完全一致。

### step4 脱离shared memory，使用warp shuffle函数
在官方文档C++ Language Extension中有一节Warp Shuffle Functions提到__shuffle_sync()系列函数可以在warp内的线程可以不借助共享内存直接互相交换一些数据  
比如这里要用到的
```C++
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);  
```
第一个参数mask，通过32个bit指定warp内参与的线程lane（从0到31）  
第二个参数var，互相要交换的值
第三个参数delta，交换的间隔。__shfl_down_sync会把执行调用的线程warp lane 加上delta，得到的目标值对应的warp lane的var返回。比如delta为16，现在warp lane为1的线程调用了这个函数，会返回warp lane为1+16=17的线程的var值。也就是17号线程把值传给了1号线程，就像在沿着warp lane向下传递一样，所以叫shfl down。注意超出了[0, warpSize - 1]的warp lane是没有意义的，调用会返回自身的var值。  
第四个参数width 默认值为warpSize是分组大小，将warp划分为若干个with大小，当lane ID + delta超过了这组with的下标则调用返回自身的var值。比如width为16，warp将按照lane分为两组0-15 16-31，当delta为2时，线程15是不能获取到15+2=17号线程的值的，因为超过了分组的边界。


于是，我们可以借助这个函数对求和过程再做一些优化。
我们现在先站在warp角度思考一下问题：每个warp内有32个线程，32个线程依次按stride读取数据相加，分别有32个结果。可以调用__shfl_down_sync把32个结果reduce到一个线程的变量上。
```C++
    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;
    float val = 0.0;
    int stride = BLOCK_SIZE * blockDim.x;
    while (idx < N) {  // grid stride loop to load data
       val += gdata[idx];
       idx += stride;
    }
    int warpLane = tid % warpSize;
    int warpId = tid / warpSize;
    unsigned int allThreadMask = 0xFFFFFFFFU;
    for(int shufOffset = 16; shufOffset > 0; shufOffset >>= 1){
        float tmp = __shfl_down_sync(allThreadMask, val, shufOffset);
        val += tmp;
    }
    // finish sum of sdata
```
然后我们再站在block的角度思考一下问题，现在有BLOCK_SIZE/warpSize = 8个warp计算出了8个值，所以需要这么大的共享内存进行warp间的通信最后累加。所以有
```C++
__global__ void reduce(float *gdata, float *out, long long N){
    __shared__ float sdata[BLOCK_SIZE/warpSize];
    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;
    float val = 0.0;
    int stride = BLOCK_SIZE * blockDim.x;
    while (idx < N) {  // grid stride loop to load data
       val += gdata[idx];
       idx += stride;
    }
    int warpLane = tid % warpSize;
    int warpId = tid / warpSize;
    unsigned int allThreadMask = 0xFFFFFFFFU;
    for(int shufOffset = warpSize/2; shufOffset > 0; shufOffset >>= 1){
        float tmp = __shfl_down_sync(allThreadMask, val, shufOffset);
        val += tmp;
    }
    if(warpLane == 0)
        sdata[warpId] = val;
    __syncthreads();    //确保不同warp的同步
    if(warpId == 0){
        for(int shufOffset = warpSize/2; shufOffset > 0; shufOffset >>= 1){//其实可以把mask和offset写的更精细 偷懒复用一下代码^_^
            val = (tid > shufOffset)?sdata[tid] : 0;
            float tmp = __shfl_down_sync(allThreadMask, val, shufOffset);
            val += tmp;
        }
        if(tid == 0)
            out[blockIdx.x] = val;
    }
}
```

### 补充说明 bank conflict
shared memory是按照4个字节大小的32个bank组织起来的，当同一个warp内的不同线程在同一条指令访问同一个bank内的不同地址时，就会发生bankconflict，会导致warp内线程不能一次访存成功。 
注意
1.不同warp的线程不存在bank conflict
2.同一个warp内一个线程先后访问不同bank不存在bank conflict，本身就已经不是一次访存了，没有冲突的说法。
3.同一个warp内不同线程同时访问相同bank同一个地址，不造成conflict。会把这次访存合并。
再次强调，只有同一个warp，不同的线程，访问同一个bank的不同内存才有冲突。比如有__shared float arr[64]; 因为float是4个字节，所以arr[0]与arr[32]同bank，与其他的元素不是同一个bank。当线程0访问arr[0]而其他线程同时访问arr[32]时，就发生了bank conflict.