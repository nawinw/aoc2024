#pragma once

#define CHECK(e) { \
  if (e != cudaSuccess) \
  { \
    printf("Error: %s %s %d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(-1); \
  } \
}
