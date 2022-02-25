/**
 * \file HipTimer.h
 * \brief A simple timer class for HIP based on events.
 *
 * \author Pierre Kestener
 * \date Feb 25 2022
 *
 */
#ifndef HIP_TIMER_H_
#define HIP_TIMER_H_

#include <hip/hip_runtime.h>

/**
 * \brief a simple timer for HIP kernel.
 */
class HipTimer
{
protected:
  hipEvent_t startEv, stopEv;
  double total_time;

public:
  HipTimer() {
    hipEventCreate(&startEv);
    hipEventCreate(&stopEv);
    total_time = 0.0;
  }

  ~HipTimer() {
    hipEventDestroy(startEv);
    hipEventDestroy(stopEv);
  }

  void start() {
    hipEventRecord(startEv, 0);
  }

  void reset() {
    total_time = 0.0;
  }

  /** stop timer and accumulate time in seconds */
  void stop() {
    float gpuTime;
    hipEventRecord(stopEv, 0);
    hipEventSynchronize(stopEv);
    hipEventElapsedTime(&gpuTime, startEv, stopEv);
    total_time += (double)1e-3*gpuTime;
  }

  /** return elapsed time in seconds (as record in total_time) */
  double elapsed() const {
    return total_time;
  }

}; // class HipTimer

#endif // HIP_TIMER_H_
