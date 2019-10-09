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

### 8. How to synchronize processes

When sharing data among processes, one need to take the responsibility to garentee that data is consistant. Here are some tips:

- Lock, for each `Lock` class, it holds two methods: `acquire()` and `release()` to control the rights to read/write.
- Event, to implement simple messages between processes. One process sends the signal and the other waits the signal. (`set()` and `clear()`)
- Condition, `wait()` and `notify_all()`
- Semaphore
- Rlock
- Barrier, to limit the processing order of processes

Here is an example on how to utilize `barrier` to synchronize two processes.

```python
import multiprocessing
from multiprocessing import Barrier, Lock, Process
from time import time
from datetime import datetime

def test_with_barrier(synchronizer, serializer):
  name = multiprocessing.current_process().name
  synchronizer.wait()
  now = time()
  with serializer:
    print("process %s ---> %s"%(name, datetime.fromtimestamp(now)))
def test_without_barrier():
  name = multiprocessing.current_process().name
  now = time()
  print("process %s ---> %s"%(name, datetime.fromtimestamp(now)))
  
if __name__ == '__main__':
  synchronizer = Barrier(2)
  serializer = Lock()
  Process(name='p1-test_with_barrier', target=test_with_barrier, args=(synchronizer, serializer)).start()
  Process(name='p2-test_with_barrier', target=test_with_barrier, args=(synchronizer, serializer)).start()
  Process(name='p3-test_without_barrier', target=test_without_barrier).start()
  Process(name='p4-test_without_barrier', target=test_without_barrier).start()
```

### 9. A simple way to share data among processes-- Manager!

When you new a manager, it can hold what you want and allows different processes to access it. 

```python
import multiprocessing
def worker(dictionary, key, item):
  dictionary[key] = item
  print("key = %d value = %d"%(key, item))
  
if __name__ == "__main__":
  mgr = multiprocessing.Manager()
  dictionary = mgr.dict()
  jobs = [multiprocessing.Process(target=worker, args=(dictionary, i, i*2)) for i in range(7)]
  for j in jobs:
    j.start()
  for j in jobs:
    j.join()
  print("Results", dictionary)
```

### 10. How to use ProcessPool

- `apply()`: block until getting result
- `apply_async()`: the return value is an object, and it's a asynchronous operation, which means, the main process will continue until all child processes start processing.
- `map()`: it can receive iteratable data and process functions in parallel.
- `map_async()`: omit

```python
import multiprocessing
def function_square(data):
  result = data**2
  return result

if __name__ == '__main__':
  inputs = list(range(100))
  pool = multiprocessing.Pool(processes=4)
  pool_outputs = pool.map(function_square, inputs)
  pool.close()
  pool.join()
```

由于MPI4Py在mac上好像装不上，而且我可能用不到，就又跳了。。。第三章结束～

## Chapter 4 Asynchronous Programming

### 1. Use `concurrent.futures` Module

This module consists of:

- `Concurrent.futures.Executor`, which is a virtual base class, and provides the method to execute asynchronize.
- `submit(function, argument)`
- `map(function, augment)`
- `shutdown(Wait=True)`

```python
import concurrent.futures
import time

num_list = [1, 2, 3, 4, 5, 6]
def evaluate_item(x):
  res = count(x)
  return result_item

def count(num):
  for i in range(0, 1000000):
    i += 1
  return i * number

if __name__ == "__main__":
  start_time = time.time()
  for item in num_list:
    print(evaluate_item(item))
    
  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(evaluate_item, item) for item in num_list]
    for future in concurrent.futures.as_complete(futures):
      print(future.result())
      
  with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(evaluate_item, item) for item in num_list]
    for future in concurrent.futures.as_complete(futures):
      print(future.result())
```

The above code implements how to utilize `ThreadPoolExecutor` and `ProcessPoolExecutor`.

### 2. Use `Asyncio` to manage events loop

`Asyncio` consists of 

- Events Loop, so each process owns their own event loop
- Coroutine(协程) is the general concepts of child process. It can be paused during processing, so that it can wait until something finished
- Futures, represents for unfinished computation
- Tasks, the child class of `Asyncio`

The first question is what is event loop?

During the process of the program, it continously trace the order of events and put them into the queue.

```python
while(1){
  events = getEvents();
  for (e in events):
  	processEvent(e);
}
```

Here is an example code

```python
import asyncio
import datetime
import time

def func_1(end_time, loop):
    print("Func1 called")
    if (loop.time() + 1.) < end_time:
        loop.call_later(1, func_2, end_time, loop)
    else:
        loop.stop()

def func_2(end_time, loop):
    print("Func2 called")
    if (loop.time() + 1.) < end_time:
        loop.call_later(1, func_3, end_time, loop)
    else:
        loop.stop()

def func_3(end_time, loop):
    print("Func3 called")
    if (loop.time() + 1.) < end_time:
        loop.call_later(1, func_1, end_time, loop)
    else:
        loop.stop()

def func_4(end_time, loop):
    print("Func4 called")
    if (loop.time() + 1.) < end_time:
        loop.call_later(1, func_4, end_time, loop)
    else:
        loop.stop()

loop = asyncio.get_event_loop()

end_loop = loop.time() + 9.
loop.call_soon(func_1, end_loop, loop)
loop.run_forever()
loop.close()
```

### Use `Asyncio` to manage Coroutines

Before we start this lesson, you need to know what is **Coroutines**. Actually, coroutine is something like child function. The difference between child function and coroutine is that child function needs to be called by the main process while coroutines executed by themselves and they are connected by channels.

```python
import asyncio

@asyncio.coroutine
def coroutine_func(func_arguments):
  # Do Something
```

Let's see how to simulate finite state machine through coroutines:

```python
import asyncio
from random import randint
import time

@asyncio.coroutine
def StartState():
    print("Start State called \n")
    input_value = randint(0, 1)
    time.sleep(1)
    if (input_value == 0):
        result = yield from State2(input_value)
    else:
        result = yield from State1(input_value)
    print("Resume of the Transition : \nStart State calling " + result)

@asyncio.coroutine
def State1(transition_value):
    outputValue =  str("State 1 with transition value = %s \n" % transition_value)
    input_value = randint(0, 1)
    time.sleep(1)
    print("...Evaluating...")
    if input_value == 0:
        result = yield from State3(input_value)
    else :
        result = yield from State2(input_value)
    result = "State 1 calling " + result
    return outputValue + str(result)

@asyncio.coroutine
def State2(transition_value):
    outputValue =  str("State 2 with transition value = %s \n" % transition_value)
    input_value = randint(0, 1)
    time.sleep(1)
    print("...Evaluating...")
    if (input_value == 0):
        result = yield from State1(input_value)
    else :
        result = yield from State3(input_value)
    result = "State 2 calling " + result
    return outputValue + str(result)

@asyncio.coroutine
def State3(transition_value):
    outputValue = str("State 3 with transition value = %s \n" % transition_value)
    input_value = randint(0, 1)
    time.sleep(1)
    print("...Evaluating...")
    if (input_value == 0):
        result = yield from State1(input_value)
    else :
        result = yield from EndState(input_value)
    result = "State 3 calling " + result
    return outputValue + str(result)

@asyncio.coroutine
def EndState(transition_value):
    outputValue = str("End state with transition value = %s \n" % transition_value)
    print("Stop computation...")
    return outputValue

loop = asyncio.get_event_loop()
loop.run_until_complete(StartState())
```

Something like DFS actually hhh.

### Use `Asyncio` to control Tasks

```python
import asyncio

@asyncio.coroutine
def factorial(num):
    f = 1
    for i in range(2, num + 1):
        print("Asyncio.Task: Compute factorial(%s)"%(i))
        yield from asyncio.sleep(1)
        f *= i
    print("Asyncio Task - factorial(%s) = %s"%(num, f))

@asyncio.coroutine
def fibonacci(num):
    a, b = 0, 1
    for i in range(num):
        print("Asyncio Task: Compute Fib (%s)"%(i))
        yield from asyncio.sleep(1)
        a, b = b, a + b
    print("Asyncio Task - fib(%s) = %s"%(num, a))

@asyncio.coroutine
def binomialCoeff(n, k):
    result = 1
    for i in range(1, k+1):
        result = result * (n-i+1) / i
        print("Asyncio Task: Compute binomialCoeff (%s)"%(i))
        yield from asyncio.sleep(1)
    print("Asyncio Task - binomialCoeff(%s, %s) = %s"%(n, k, result))

tasks = [asyncio.Task(factorial(10)),
         asyncio.Task(fibonacci(10)),
         asyncio.Task(binomialCoeff(20, 10))]
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
```

So you can view this program as something that an execute line skip among three functions(actually coroutines). When we called `yield`, we will move to the next task.

And if you don't call `yield`, you may find these tasks are executed one by one orderly.



## Reference

[1] <a href="https://python-parallel-programmning-cookbook.readthedocs.io/zh_CN/latest/index.html">Python并行编程 中文版</a>





