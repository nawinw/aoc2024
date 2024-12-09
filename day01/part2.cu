#include <cuda/std/utility>
#include <cuda/std/cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "utility.h"

template<typename T>
__global__ void similarity(T* a, T* b, T* out, size_t len)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
  {
    int count = 0;
    for (int j = 0; j < len; j++)
    {
      if (a[i] == b[j])
        count++;
    }
    out[i] = a[i] * count;
  }
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
  int *result_gpu{nullptr};
  CHECK(cudaMalloc(&list1_gpu, list1.size() * sizeof(list1[0])));
  CHECK(cudaMalloc(&list2_gpu, list2.size() * sizeof(list2[0])));
  CHECK(cudaMalloc(&result_gpu, list2.size() * sizeof(list2[0])));
  CHECK(cudaMemcpy(list1_gpu, list1.data(), list1.size() * sizeof(list1[0]), cudaMemcpyDefault));
  CHECK(cudaMemcpy(list2_gpu, list2.data(), list2.size() * sizeof(list2[0]), cudaMemcpyDefault));

  int NUM_THREADS = 16;
  dim3 NUM_BLOCKS(list1.size() / NUM_THREADS + 1);
  similarity<<< NUM_THREADS, NUM_BLOCKS >>>(list1_gpu, list2_gpu, result_gpu, list1.size());
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(list1.data(), result_gpu, list1.size() * sizeof(list1[0]), cudaMemcpyDefault));

  uint64_t sum = 0;
  for (const auto& item: list1)
    sum += item;
  std::cout << "Answer: " << sum << std::endl;
}