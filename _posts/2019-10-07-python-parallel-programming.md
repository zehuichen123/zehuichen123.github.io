---
layout: default
title: Python Parallel Programming
date: 2019-10-07 18:39 +0800
---

## Chapter 1 Intro to Python and Parallel Computing

### 1. Computer system framework

SISD(I=instruction, D=data), SIMD, MISD, MIMD

### 2. Memory Management

The way to get data, in other words, the time of processor to access memory data.

Actually, there are two different categories:

- Shared Memory
- Distrbuted Memory

#### Shared Memory

All processors share the same logic memory address.

There are actually 4 ways to **access memory** under shared memory framework:

- UMA(uniform memory access)
- NUMA(non-uniform memory access), it can be divided into high speed memory access area and low speed memory access area. (local cache may refer to high speed)
- NORMA(no remote memory access), only has local cache. To access other memory, one need to communicate with other processors.
- COMA(cache only memory access)

#### Distributed Memory

Each processor has its own physical memory and own logic memory address, they are individual.

We may mention some applications of distributed memory.

- MPP(massively parallel processing)
- Cluster(fail-over cluster, load balancing cluster, high-performance cluster)

**Heterogeneous architecture**, CPU manipulates tasks and split the original work into several single and highly-parallel tasks and assigns them to GPU for high speed processing.

### 3. Parallel Programming Model

- Shared Memory Model, all processors shared one memory. Computer utilizes locks and signal to control conflicts on read and write.
- Multi-threading Model, one processor can have multiple tasks concurrently. Actually, this is implemented through time slice.
- Message Passing Model, widely used in distributed memory system.
- Data-Parallel Model, one need to specify the assignment of data and their alignment.

### 4. How to design parallel program

- **Tasks decomposition**
  - Domain Level, data may be decomposed. Maybe like the data-parallel model, IMO.
  - Functional Level, split the task into smaller one.
- **Tasks Assignment**
  - **Load balancing**
  - Messaging between processors
- **Polymerization**(聚合), in order to reduce the cost of messaging between each processor.
- **Mapping**, assign different tasks to different processors, for instance, the tasks which need to message frequently may be assigned to one processor to improve locality.
- **Dynamic Mapping**, some local mapping algorithms can be better than global ones since they can find the local optimal based on current situations.

### 5. How to judge the performance of one parallel program

Omitted... Refer to <a href="https://python-parallel-programmning-cookbook.readthedocs.io/zh_CN/latest/chapter1/06_How_to_evaluate_the_performance_of_a_parallel_program.html">6. 如何评估并行程序的性能</a>

### 6,7,8

Omitted... You may refer to <a href="https://www.liaoxuefeng.com/wiki/1016959663602400">here(廖雪峰的python教程</a>) to learn Python

### 9. Intro to Process and Threading

You have to remember one sentence and that's enough:

> 进程是系统进行资源分配和调度的一个独立单位. 线程是进程的一个实体,是CPU调度和分派的基本单位,它是比进程更小的能独立运行的基本单位.线程自己基本上不拥有系统资源,只拥有一点在运行中必不可少的资源。

## Chapter 2 Parallel based on Threading

考虑到Python目前我需要用到的加速都是基于进程的，打算跳过这章，先学第三章，以后有空再回来学这章。至于为啥嘞，首先Python有GIL，所以不论你开几个线程，默认Python都只会占用一个核，可是我们现在基本上都是多核处理器了，所以想要提高效率就要增加每个CPU的使用率。其次，线程相对于进程的一个比较大的优势就是它的上下文切换速度比进程快，但是实际上利用多个CPU并行的提升远远大于上下文切换的代价~没错 我就是这么功利：）

## Chapter 3 Parallel based on Process

### 1. Intro

In this chapter, we will mainly cover about `multiprocessing` and `mp4py` module in Python library.

`multiprocessing` implements the shared memory mechanism which enables processors to access shared memory.

`mp4py` implements message passing mechanism (design pattern) which enables processes messages without sharing anything. All information are passed through messages.

Here is one example code

```python
from multiprocessing import Pool
def f(x):
  return x * x
p = Pool(5)
p.map(f, [1, 2, 3])

>>output: [1, 4, 9]
```

The author(maybe translator) mentioned that function f should be declared out of this file and called as one module. However, that's not necessary. What you need to do is to define function `f` before you declare process pool, like what I wrote above.

### 2. How to generate one process?

Spawn means generate, which refers to the generation of son process by its father process. These process can be executed asynchronous or synchronous. The following code shows how can we create processes:

```python
import multiprocessing
def foo(i):
  print("called function in process: %s" % i)
  return
if __name__ == '__main__':
  process_jobs = []
  for i in range(5):
    p = multiprocessing.Process(target=foo, args=(i,))
    process_jobs.append(p)
    p.start()
    p.join()
```

`start` means start this process and `join` means wait until the process finished.

If you forget to call `join` method, the process will not be freed, even if the main process ends.

### 3. How to name one process

Omitted... No use for now

### 4. How to run process in the background

Set the **daemon** to be True.

```python
import multiprocessing
import time

def foo():
  name = multiprocessing.current_process().name
  print("Starting>", name)
  time.sleep(3)
  print("Exiting %s"% name)
  
if __name__ == '__main__':
  bg_process = multiprocessing.Process(name='bg_process', target=foo)
  bg_process.daemon = True
  NO_bg_process = multiprocessing.Process(name='NO_bg_process', target=foo)
  NO_bg_process.daemon = False
  bg_process.start()
  NO_bg_process.start()
```

background process is not allowed to new more son process. Otherwise, its child process may become orphan process when background process exit along with its father process.

### 5. How to terminate process

```python
import multiprocessing
import time
def foo():
  print('Starting function')
  time.sleep(0.1)
  print('Finished function')
if __name__ == '__main__':
  p = multiprocessing.Process(target=foo)
  p.start()
  p.terminate()
  p.join()
```

### 7. How to exchange object within processes

```python
import multiprocessing
import time
import random

class Producer(multiprocessing.Process):
  def __init__(self, queue):
    super().__init__()
    self.queue = queue
  def run(self):
    for i in range(10):
      item = random.randint(0, 256)
      self.queue.put(item)
      print("Process Producer : item %d appended to queue %s" % (item, self.name))
      time.sleep(1)
      print("The size of queue is %s" % self.queue.qsize())
class Consumer(multiprocessing.Process):
  def __init__(self, queue):
    multiprocessing.Process.__init__(self)
    self.queue = queue
  def run(self):
    while True:
      if self.queue.empty():
        print("the queue is empty")
        break
      else:
        time.sleep(2)
        item = self.queue.get()
        print('Process Consumer : item %d popped from by %s \n' % (item,self.name))
        time.sleep(1)
if __name__ == '__main__':
  queue = multiprocessing.Queue()
  process_producer = Producer(queue)
  process_consumer = Consumer(queue)
  process_producer.start()
  process_consumer.start()
  process_producer.join()
  process_consumer.join()
```



## Reference

[1] <a href="https://python-parallel-programmning-cookbook.readthedocs.io/zh_CN/latest/index.html">Python并行编程 中文版</a>





