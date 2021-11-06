# 异构框架的选择

- 如何训练一个模型? 没有脚手架,门槛会很高,你会很痛苦
  1. 设计模型,推导公式
  2. 编写算子程序(GPU,CPU)
  3. 基于算子实现反向传播算法
  4. 编写UT测试正确性
  5. 优化执行效率
  6. 开始做实验调参数
  7. 更多事情。。。

- 工业级不可缺少的内容

  1. 大数据：工业界的优势
  2. 强算力：GPU
  3. 算法框架

  

## 流行的深度学习异构框架

1. `Caffe` (2013)针对卷积神经网络设计的异构框架，通过堆叠layer搭建神经网络和training的流水线，但对于搭建NLP这类复杂的神经网络就显得不方便了。
2. `TensorFlow`、`mxnet`(2015)对比caffe的区别：
   - 细粒度更高：有layer级别降为OP级别
   - 结构化、符号化编程：引入自动求导
   - 静态图机制

3. `PaddlePaddle`(2016)百度自研深度学习框架
4. `Caffe2` (2017)
5. `Pythorch`(2018)
   - 去除图的机制

![image-20211106134436346](Assets/TensorFlow框架原理/image-20211106134436346.png)

6. `Tensorflow2.0`融入了 Eager模式,并正在通过推广 `Keras`约束API



## 宏观初探TF框架

![image-20211106134441758](Assets/TensorFlow框架原理/image-20211106134441758.png)



# 环境准备与TF生态



## GPU和CUDA

- GPU
  1. 专用计算设备
  2. 大量计算core,并发度极高 
  3. 造价非常昂贵
- CPU
  1. 负责计算和控制
  2. 并发度远不如GPU
  3. 造价比GPU低
- CUDA
  NVIDIA推出的GPU并行编程架构
  1. 语法属于扩展的C语言
  2. 编写高质量程序需要相当程度地了解GPU的架构和硬件特性
  3. 调试门槛较高
  4. 并行编程思想



# 静态图机制

模型定义与运行分离的机制



## 编程模式抽象与Graph

类似于编译器，在TensorFlow层面将模型抽象成有向无环图DAG.

- Operation(`Op`)
  一种细粒度运算的抽象表示,比如add, matmul, concat等等。
- `Kernel`
  某种Op在特定 device(CPU,GPU)上的实现。有些Op可能只有一种 device的实现。



### 计算图的表示——Graph

用Op搭建模型的过程，实际上是在组装Graph，并没有被运行.

![image-20211106134232839](Assets/TensorFlow框架原理/image-20211106134232839.png)















## 驱动Graph运行的Client——Session































