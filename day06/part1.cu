#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "utility.h"

template<typename T1, typename T2>
__global__ void xmas_checker(T1* data, T2* result, int rows, int cols, int stride)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (!(i < rows && j < cols))
    return;

  const char *XMAS = "XMAS";

  T2 count = 0;
  for (int di = -1; di < 2; di++)
  {
    for (int dj = -1; dj < 2; dj++)
    {
      bool is_xmas = true;
      for (int idx = 0; idx < 4; idx++)
      {
        int new_i = i + di * idx;
        int new_j = j + dj * idx;
        bool coord_valid = new_i >= 0 && new_i < rows && new_j >= 0 && new_j < cols;
        is_xmas = is_xmas && coord_valid && XMAS[idx] == data[new_i * stride + new_j];
      }
      if (is_xmas)
        count++;
    }
  }
  result[i * cols + j] = count;
}

int main()
{
  std::stringstream ss;
  std::ifstream ifs("../day04/part1.txt");
  ss << ifs.rdbuf();
  std::string str = ss.str();

  int cols = str.find('\n');
  int stride = cols + 1;
  int rows = (str.size() + 1) / stride;

  std::cout << rows << ", " << cols << ", " << stride << std::endl;
  std::vector<int> result(rows * cols);

  char *data_gpu{nullptr};
  int *result_gpu{nullptr};
  CHECK(cudaMalloc(&data_gpu, str.size()));
  CHECK(cudaMalloc(&result_gpu, result.size() * sizeof(result[0])));
  CHECK(cudaMemcpy(data_gpu, str.data(), str.size() * sizeof(str[0]), cudaMemcpyDefault));

  dim3 num_threads(1024, 1024);
  dim3 num_blocks( cols / num_threads.x + 1, rows / num_threads.y + 1);
  xmas_checker<<<num_threads, num_blocks>>>(data_gpu, result_gpu, rows, cols, stride);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(result.data(), result_gpu, result.size() * sizeof(result[0]), cudaMemcpyDefault));

  uint64_t sum = 0;
  for (const auto& n: result)
  {
    //std::cout << n << ", ";
    sum += n;
  }
  std::cout << "Answer: " << sum << std::endl;
}
