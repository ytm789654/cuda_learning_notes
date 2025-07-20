### 同步和异步
同步调用是指等待函数执行结束再返回，异步调用是直接返回一个预设的结果，函数执行在另一个线程中，结果并不马上返回给调用者。<br>
也可以认为同步调用会阻塞程序向下执行，而异步调用不会阻塞。

### CUDA中的异步操作
▶ Computation on the host;<br>
▶ Computation on the device;<br>
▶ Memory transfers from the host to the device;<br>
▶ Memory transfers from the device to the host;<br>
▶ Memory transfers within the memory of a given device;<br>
▶ Memory transfers among devices.<br>

其中从host视角看，这些操作是异步的<br>
▶ Kernel launches;<br>
▶ Memory copies within a single device’s memory;<br>
▶ Memory copies from host to device of a memory block of 64 KB or less;<br>
▶ Memory copies performed by functions that are suffixed with Async;<br>
▶ Memory set function calls.<br>
也就是说从device到host拷贝内存会阻塞host线程。

而一个典型的CUDA程序一般是这么几步
1.host准备数据<br>
2.把数据从host拷贝到device（异步）<br>
3.launch kernel完成计算<br>
4.从device将结果拷贝回host<br>
既然各种操作都是异步的，那么执行顺序是不能保证的。234的先后顺序是如何保证的？<br>

实际上各种在device上执行的命令先后顺序是可以通过stream控制的，如果不特意指示，所有命令都会通过默认的stream执行，这样就保证了先后顺序，虽然在host的视角看，拷贝内存，launch kernel都是异步的，但是在对应的stream上，这些行为都是同步的。
### device上