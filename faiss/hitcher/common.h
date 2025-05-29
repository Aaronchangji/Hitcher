#ifndef HITCHER_COMMON_H
#define HITCHER_COMMON_H

#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_priority_queue.h>
#include <tbb/concurrent_map.h>
#include "concurrentqueue.h"
#include <random>
#include <chrono>
#include <atomic>
#include <iostream>

namespace faiss{

namespace hitcher {

constexpr size_t kNumVecPerCluster = 512;

constexpr float kGPUMemForCacheRatio = 0.99;

constexpr size_t kGPUTmpMemSizeTop = (size_t)64 * 1024 * 1024;

constexpr size_t kGPUTmpMemSizeBottom = (size_t)256 * 1024 * 1024;

template <typename T>
using ConcurrentQueue = tbb::concurrent_bounded_queue<T>;

template <typename T1, typename T2>
using ConcurrentPriorityQueue = tbb::concurrent_priority_queue<T1, T2>;

template <typename T>
using FastConcurrentQueue = moodycamel::ConcurrentQueue<T>;

template <typename T1, typename T2>
using ConcurrentMultiMap = tbb::concurrent_multimap<T1, T2>;

class SpinLock {
  private:
    std::atomic<bool> flag_;

  public:
    SpinLock() : flag_(false) {}

    void lock() {
      bool expect = false;
      while (!flag_.compare_exchange_weak(expect, true)) {
        //这里一定要将expect复原，执行失败时expect结果是未定的
        expect = false;
      }
    }

    bool tryLock() {
      bool expect = false;
      if (!flag_.compare_exchange_weak(expect, true)) {
        return false;
      }
      return true;
    }

    void unlock() {
      flag_.store(false);
    }
};

using PerfClock = std::chrono::high_resolution_clock;

static auto ScheduleDistribution(double qps) {
  return [dist = std::exponential_distribution<>(qps)](auto& gen) mutable {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(dist(gen)));
  };
}

static void bindCore(uint16_t core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    assert(false);
  }
}

template<typename T>
static void print_vec_info(std::vector<T>& vec, std::string info, std::string unit) {
  if (vec.size() < 100) printf("No enough items in vector.\n");

  std::sort(vec.begin(), vec.end());
  int p50_idx = int(vec.size() * 0.5);
  int p95_idx = int(vec.size() * 0.95);
  int p99_idx = int(vec.size() * 0.99);
  T p50_value = vec[p50_idx];
  T p95_value = vec[p95_idx];
  T p99_value = vec[p99_idx];
  std::cout << info << ": P50:" << p50_value << unit << ", P95: " << p95_value << unit << ", P99: " << p99_value << unit << std::endl;
  fflush(stdout);
}

}

}

#endif