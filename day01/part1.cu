#include <cuda/std/utility>
#include <cuda/std/cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "utility.h"

template<typename T>
__global__ void quicksort(T* start, size_t len)
{
  T pivot = start[len - 1];
  size_t low = 0;
  for (size_t i = 0; i < len; i++)
  {
      if (start[i] <= pivot)
        cuda::std::swap(start[low++], start[i]);
  }

  if (low >= 2)
  {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    quicksort<<<1, 1, 0, s>>>(start, low - 1);
    cudaStreamDestroy(s);
  }

  if (len - low > 1)
  {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    quicksort<<<1, 1, 0, s>>>(start + low, len - low);
    cudaStreamDestroy(s);
  }
}

template<typename T>
__global__ void dist(T* a, T* b, size_t len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
    a[i] = cuda::std::abs(a[i] - b[i]);
}

int main()
{
  std::vector<int> list1, list2;
  std::ifstream ifs("../day01/part1.txt");
  while(ifs.good())
  {
    int a, b;
    ifs >> a >> b;
    list1.push_back(a);
    list2.push_back(b);
  }

  int *list1_gpu{nullptr};
  int *list2_gpu{nullptr};
  CHECK(cudaMalloc(&list1_gpu, list1.size() * sizeof(list1[0])));
  CHECK(cudaMalloc(&list2_gpu, list2.size() * sizeof(list2[0])));
  CHECK(cudaMemcpy(list1_gpu, list1.data(), list1.size() * sizeof(list1[0]), cudaMemcpyDefault));
  CHECK(cudaMemcpy(list2_gpu, list2.data(), list2.size() * sizeof(list2[0]), cudaMemcpyDefault));

  quicksort<<< 1, 1 >>>(list1_gpu, list1.size());
  CHECK(cudaGetLastError());

  quicksort<<< 1, 1 >>>(list2_gpu, list2.size());
  CHECK(cudaGetLastError());

  int NUM_THREADS = 16;
  dim3 NUM_BLOCKS(list1.size() / NUM_THREADS + 1);
  dist<<< NUM_THREADS, NUM_BLOCKS >>>(list1_gpu, list2_gpu, list1.size());
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(list1.data(), list1_gpu, list1.size() * sizeof(list1[0]), cudaMemcpyDefault));
  CHECK(cudaMemcpy(list2.data(), list2_gpu, list2.size() * sizeof(list2[0]), cudaMemcpyDefault));

  uint64_t sum = 0;
  for (const auto& item: list1)
    sum += item;
  std::cout << "Answer: " << sum << std::endl;
}