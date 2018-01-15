#ifndef JACOBIBARRIER_H
#define JACOBIBARRIER_H

#include <atomic>
#include <functional>
#include <string.h>
#include <unistd.h>

class JacobiBarrier {
private:
  const int threads_num;

public:
  std::atomic<int> count;
  std::atomic<int> generation;

  JacobiBarrier(int n) : threads_num(n), count(0), generation(0) {}  

  /** Signals that the calling thread has arrived at the synchronization point
    * and block until all the participating threads have arrived.
   **/
  bool arrive_and_wait(std::function<void()> fun) {
    /** Atomic operations tagged memory_order_relaxed are not synchronization
     *  operations; they do not impose an order among concurrent memory
     * accesses.
     *  They only guarantee atomicity and modification order consistency.
    */
    int my_gen = generation.load(); // memory_order_relaxed

    if (count.fetch_add(1) == threads_num - 1) {
      fun();
      count.store(0);
      generation.fetch_add(1); // memory_order_relaxed
      return true;
    } else {
      do {
      } while (my_gen == generation.load()); // wait
      return false;
    }
  }
};

#endif