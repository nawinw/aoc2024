#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include "utility.h"

std::map<std::pair<uint64_t, int>, uint64_t> memo;

uint64_t blink(uint64_t n, int blinks)
{
  auto index = std::make_pair(n, blinks);
  if (memo.find(index) != memo.end())
    return memo[index];

  uint64_t count = 1;
  while(blinks > 0)
  {
    blinks--;
    int num_digits = floor(log((double)n) / log(10.0)) + 1;
    if (n == 0)
    {
      n = 1;
    }
    else if ((num_digits % 2) == 0)
    {
      uint64_t divisor = pow(10.0, num_digits / 2);
      uint64_t n1 = n % divisor;
      uint64_t n2 = n / divisor;

      count = blink(n1, blinks) + blink(n2, blinks);
      break;
    }
    else
    {
      n *= 2024;
    }
  }

  memo[index] = count;
  return count;
}


int main()
{
  std::vector<int> data={5910927, 0, 1, 47, 261223, 94788, 545, 7771};

  uint64_t count = 0;
  for(const auto& datum: data)
    count += blink(datum, 75);

  std::cout << "Answer: " << count << std::endl;
}
