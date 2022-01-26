#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_METRIC_SPACE_DATAPOINT_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_METRIC_SPACE_DATAPOINT_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "external/gbbs/gbbs/gbbs.h"

namespace research_graph {
namespace in_memory {

struct DataPoint {
  std::vector<float> coordinates;

  DataPoint() {}
  explicit DataPoint(const std::vector<float>& coordinates)
    : coordinates(coordinates) {}
};

}  // namespace in_memory
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_METRIC_SPACE_DATAPOINT_H_