#include <cuda/std/cmath>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "utility.h"

template<typename T>
__global__ void checker(T* data, size_t rows, size_t cols, size_t stride)
{
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < cols)
  {
    int omit = -1;
    T result = 0;
    do
    {
      T increasing = 1;
      T decreasing = 1;
      T has_small_diffs = 1;
      T prev;
      bool init = true;
      size_t i = 0;
      while(data[i * stride + j] != -1)
      {
        if (i != omit)
        {
          if (init)
          {            
            prev = data[i * stride + j];
            init = false;
          }
          else
          {
            T cur = data[i * stride + j];
            has_small_diffs = has_small_diffs & (cuda::std::abs(prev - cur) <= 3);
            increasing = increasing & (cur > prev);
            decreasing = decreasing & (cur < prev);
            prev = cur;
          }
        }
        i++;
      }
      __syncthreads(); // ?????
      result = result | ((increasing ^ decreasing) & has_small_diffs);
      omit++;
    } while(data[omit * stride + j] != -1);
    data[j] = result;
  }
}

int main()
{
  const size_t rows = 256;
  const size_t cols = 1024;
  std::vector<int> data;
  data.resize(rows * cols);
  size_t num_cols = 0;

#if 0
  std::ifstream ifs("../day02/example.txt");
#else
  std::ifstream ifs("../day02/part1.txt");
#endif
  while(ifs.good())
  {
    std::string line;
    std::getline(ifs, line);
    std::string token;

    auto ss = std::stringstream(line);
    size_t i = 0;
    while(std::getline(ss, token, ' '))
    {
      data[i++ * cols + num_cols] = std::stoi(token);
      assert(i < rows);
    }
    data[i * cols + num_cols++] = -1;
    assert(num_cols <= cols);
  }

  int *data_gpu{nullptr};
  CHECK(cudaMalloc(&data_gpu, data.size() * sizeof(data[0])));
  CHECK(cudaMemcpy(data_gpu, data.data(), data.size() * sizeof(data[0]), cudaMemcpyDefault));

  int NUM_THREADS = 16;
  dim3 num_blocks(num_cols / NUM_THREADS + 1);
  checker<<<NUM_THREADS, num_blocks>>>(data_gpu, rows, num_cols, cols);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(data.data(), data_gpu, data.size() * sizeof(data[0]), cudaMemcpyDefault));

  for (size_t i = 0; i < 1; i++)
  {
    for (size_t j = 0; j < num_cols; j++)
    {
      std::cout << data[i * cols + j] << ", ";
    }
    std::cout << std::endl;
  }

  int sum = 0;
  for (size_t j = 0; j < num_cols; j++)
    sum += data[j];

  std::cout << "Answer: " << sum << std::endl;
}
