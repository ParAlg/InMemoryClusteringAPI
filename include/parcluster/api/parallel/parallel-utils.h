// This file contains the definitions of parallel primitives and utilities
// which exploit multi-core parallelism.

#ifndef RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_UTILS_H_
#define RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_UTILS_H_

#include <functional>

#include "gbbs/bridge.h"

namespace research_graph {
namespace parallel {

// Forks two sub-calls that do not return any data. Each sub-call is run in a
// separate fiber. The function is synchronous and will return only after both
// left() and right() complete. If do_parallel is false, it first evaluates
// left() and then right().
static inline void ParDo(const std::function<void()>& left,
                         const std::function<void()>& right,
                         bool do_parallel = true) {
  if (do_parallel) {
    parlay::par_do(left, right);
  } else {
    left();
    right();
  }
}

// Runs f(start), ..., f(end) in parallel by recursively splitting the loop
// using par_do(). The function is synchronous and returns only after all loop
// iterations complete. The recursion stops and uses a sequential loop once the
// number of iterations in the subproblem becomes <= granularity.
static inline void ParallelForSplitting(size_t start, size_t end,
                                        size_t granularity,
                                        const std::function<void(size_t)>& f) {
  parlay::parallel_for(start, end, f, granularity);
}

}  // namespace parallel
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_UTILS_H_
