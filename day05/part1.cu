#include <cuda/std/utility>
#include <cuda/std/cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "utility.h"

const size_t ORDER_DIM = 100;

__global__ void validate(int* data, int *result, int* order, size_t rows, size_t cols, size_t stride)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j >= cols)
    return;

  int i1 = 0;
  bool valid = 1;
  while(data[i1 * stride + j] != -1)
  {
    int i2 = i1 + 1;
    while(data[i2 * stride + j] != -1)
    {
      int order_i = data[i2 * stride + j];
      int order_j = data[i1 * stride + j];
      if (order[order_i * ORDER_DIM + order_j])
        valid = 0;
      i2++;
    }
    i1++;
  }

  result[j] = valid * data[i1 / 2 * stride + j];
}

int main()
{
  std::vector<int> order(ORDER_DIM * ORDER_DIM);

  const size_t ROWS = 256;
  const size_t STRIDE = 1024;
  size_t cols = 0;
  std::vector<int> data(ROWS * STRIDE);

  std::ifstream ifs("../day05/part1.txt");
  while(ifs.good())
  {
    std::string line;
    std::getline(ifs, line);

    if (line.find("|") != std::string::npos)
    {
      int i = std::stoi(line.substr(0, 2));
      int j = std::stoi(line.substr(3, 2));
      //std::cout << i << "|" << j << std::endl;
      order[i * ORDER_DIM + j] = 1;
    }
    else if (!line.empty())
    {
      std::string token;
      auto ss = std::stringstream(line);
      size_t i = 0;
      while(std::getline(ss, token, ','))
      {
        //std::cout << token << ", ";
        data[i++ * STRIDE + cols] = std::stoi(token);
      }
      data[i * STRIDE + cols] = -1;
      cols++;
      //std::cout << std::endl;
    }
  }

  int *data_gpu{nullptr};
  int *order_gpu{nullptr};
  int *result_gpu{nullptr};
  CHECK(cudaMalloc(&data_gpu, data.size() * sizeof(data[0])));
  CHECK(cudaMalloc(&order_gpu, order.size() * sizeof(order[0])));
  CHECK(cudaMalloc(&result_gpu, cols * sizeof(result_gpu[0])));
  CHECK(cudaMemcpy(data_gpu, data.data(), data.size() * sizeof(data[0]), cudaMemcpyDefault));
  CHECK(cudaMemcpy(order_gpu, order.data(), order.size() * sizeof(order[0]), cudaMemcpyDefault));

  const int NUM_THREADS = 16;
  dim3 num_blocks(cols / NUM_THREADS + 1);
  validate<<<NUM_THREADS, num_blocks>>>(data_gpu, result_gpu, order_gpu, ROWS, cols, STRIDE);
  CHECK(cudaGetLastError());

  std::vector<int> result(cols); 
  CHECK(cudaMemcpy(result.data(), result_gpu, result.size() * sizeof(result[0]), cudaMemcpyDefault));

  uint64_t sum = 0;
  for (const auto& n: result)
    sum += n;

  std::cout << "Answer: " << sum << std::endl;
}