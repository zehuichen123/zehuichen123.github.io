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

## Reference

[1] <a href="https://python-parallel-programmning-cookbook.readthedocs.io/zh_CN/latest/index.html">Python并行编程 中文版</a>





