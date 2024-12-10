#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "utility.h"

enum State
{
  INIT,
  M,
  U,
  L,
  PAREN1,
  NUMBER1,
  COMMA,
  NUMBER2
};

template<typename T1, typename T2>
__global__ void mul_exec(T1* data, T2* result, size_t len)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= len)
    return;

  State state = INIT;
  T2 number1 = 0;
  T2 number2 = 0;
  size_t digit = 0;

  result[i] = 0;
  while (i < len)
  {
    switch(state)
    {
      case INIT:
        if (data[i] == 'm')
          state = M;
        else
          return;
        break;

      case M:
        if (data[i] == 'u')
          state = U;
        else
          return;
        break;

      case U:
        if (data[i] == 'l')
          state = L;
        else
          return;
        break;

      case L:
        if (data[i] == '(')
          state = PAREN1;
        else
          return;
        break;

      case PAREN1:
        if (data[i] >= '0' && data[i] <= '9')
        {
          number1 = number1 * 10 + (data[i] - '0');
          digit++;
        }
        else if (digit > 0 && data[i] == ',')
        {
          digit = 0;
          state = COMMA;
        }
        else
          return;
        break;

      case COMMA:
        if (data[i] >= '0' && data[i] <= '9')
        {
          number2 = number2 * 10 + (data[i] - '0');
          digit++;
        }
        else if (digit > 0 && data[i] == ')')
        {
          result[i] = number1 * number2;
          return;
        }
        else
          return;
        break;

      default:
        return;
    }
    i++;
  }
}

int main()
{
  std::stringstream ss;
  std::ifstream ifs("../day03/part1.txt");
  ss << ifs.rdbuf();
  std::string str = ss.str();
  std::vector<int64_t> result(str.size());

  char *data_gpu{nullptr};
  int64_t *result_gpu{nullptr};
  CHECK(cudaMalloc(&data_gpu, str.size()));
  CHECK(cudaMalloc(&result_gpu, result.size() * sizeof(result[0])));
  CHECK(cudaMemcpy(data_gpu, str.data(), str.size() * sizeof(str[0]), cudaMemcpyDefault));

  int NUM_THREADS = 1024;
  dim3 num_blocks(str.size() / NUM_THREADS + 1);
  mul_exec<<<NUM_THREADS, num_blocks>>>(data_gpu, result_gpu, str.size());
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy(result.data(), result_gpu, result.size() * sizeof(result[0]), cudaMemcpyDefault));

  int64_t sum = 0;
  for (auto& n: result)
  {
    //std::cout << n << ", ";
    sum += n;
  }

  std::cout << "Answer: " << sum << std::endl;
}
