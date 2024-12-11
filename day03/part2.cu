#include <cassert>
#include <fstream>
#include <limits.h>
#include <iostream>
#include <sstream>
#include <vector>
#include "utility.h"

#define DO_VAL   9000000
#define DONT_VAL 8999999

enum State
{
  INIT,
  M,
  U,
  L,
  PAREN,
  NUMBER1,
  COMMA,
  NUMBER2,
  D,
  O,
  N,
  APOSTROPHE,
  T,
  DO_PAREN,
  DONT_PAREN
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
        else if (data[i] == 'd')
          state = D;
        else
          return;
        break;

      case D:
        if (data[i] == 'o')
          state = O;
        else
          return;
        break;

      case O:
        if (data[i] == '(')
          state = DO_PAREN;
        else if (data[i] == 'n')
          state = N;
        else
          return;
        break;

      case N:
        if (data[i] == '\'')
          state = APOSTROPHE;
        else
          return;
        break;

      case APOSTROPHE:
        if (data[i] == 't')
          state = T;
        else
          return;
        break;

      case T:
        if (data[i] == '(')
          state = DONT_PAREN;
        else
          return;
        break;

      case DO_PAREN:
        if (data[i] == ')')
        {
          result[i] = DO_VAL;
          return;
        }
        else
          return;
        break;

      case DONT_PAREN:
        if (data[i] == ')')
        {
          result[i] = DONT_VAL;
          return;
        }
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
          state = PAREN;
        else
          return;
        break;

      case PAREN:
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
  bool doing = true;
  for (auto& n: result)
  {
    //std::cout << n << ", ";
    if (n == DO_VAL)
      doing = true;
    else if (n == DONT_VAL)
      doing = false;
    else if (doing)
      sum += n;
  }

  std::cout << "Answer: " << sum << std::endl;
}
