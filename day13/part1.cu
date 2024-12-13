#include <fstream>
#include <iostream>
#include <regex>
#include <vector>
#include "utility.h"
#include <limits.h>

struct Machine
{
  int2 a;
  int2 b;
  int2 prize;
};

__global__ void calc_tokens(Machine* machines, int* result, size_t len)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len)
    return;

  Machine m = machines[idx];
  int min_cost = INT_MAX;
  int answer = 0;
  for (int i = 0; i <= 100; i++)
  {
    for (int j = 0; j <= 100; j++)
    {
      int x = m.a.x * i + m.b.x * j;
      int y = m.a.y * i + m.b.y * j;
      int cost = i*3 + j;
      if (m.prize.x == x && m.prize.y == y && cost < min_cost)
        answer = min_cost = cost;
    }
  }

  result[idx] = answer;
}

int main()
{
  std::stringstream ss;
  std::ifstream ifs("../day13/part1.txt");
  std::vector<Machine> machines;
  std::regex num_regex("[0-9]+");
  while (ifs.good())
  {
    std::string line, tmp;
    std::getline(ifs, tmp); line += tmp;
    std::getline(ifs, tmp); line += tmp;
    std::getline(ifs, tmp); line += tmp;
    std::getline(ifs, tmp); line += tmp;

    Machine m;

    auto iter = std::sregex_iterator(line.begin(), line.end(), num_regex);
    m.a.x = std::stoi((iter++)->str());
    m.a.y = std::stoi((iter++)->str());
    m.b.x = std::stoi((iter++)->str());
    m.b.y = std::stoi((iter++)->str());
    m.prize.x = std::stoi((iter++)->str());
    m.prize.y = std::stoi(iter->str());
    machines.push_back(m);

    // std::cout << "Button A: X+" << m.a.x << ", Y+" << m.a.y << std::endl;
    // std::cout << "Button B: X+" << m.b.x << ", Y+" << m.b.y << std::endl;
    // std::cout << "Prize: X=" << m.prize.x << ", X=" << m.prize.y << std::endl << std::endl;
  }

  Machine *machines_gpu{nullptr};
  int *result_gpu{nullptr};
  CHECK(cudaMalloc(&machines_gpu, machines.size() * sizeof(machines_gpu[0])));
  CHECK(cudaMalloc(&result_gpu, machines.size() * sizeof(result_gpu[0])));
  CHECK(cudaMemcpy(machines_gpu, machines.data(), machines.size() * sizeof(machines[0]), cudaMemcpyDefault));

  int num_threads(16);
  dim3 num_blocks(machines.size() / num_threads + 1);
  calc_tokens<<<num_threads, num_blocks>>>(machines_gpu, result_gpu, machines.size());
  CHECK(cudaGetLastError());

  std::vector<int> result(machines.size());
  CHECK(cudaMemcpy(result.data(), result_gpu, result.size() * sizeof(result[0]), cudaMemcpyDefault));

  uint64_t sum = 0;
  for (const auto& n: result)
  {
    //std::cout << n << ", ";
    sum += n;
  }
  std::cout << "Answer: " << sum << std::endl;
}
