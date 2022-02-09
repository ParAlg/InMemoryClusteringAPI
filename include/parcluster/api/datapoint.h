#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_METRIC_SPACE_DATAPOINT_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_METRIC_SPACE_DATAPOINT_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "gbbs/gbbs.h"

namespace research_graph {
namespace in_memory {

struct DataPoint {
  absl::Span<const float> coordinates;
  absl::Span<const uint64_t> indices;

  DataPoint() {}
  explicit DataPoint(absl::Span<const float> coordinates)
    : coordinates(coordinates) {}
  DataPoint(absl::Span<const float> coordinates, absl::Span<uint64_t> indices)
    : coordinates(coordinates), indices(indices) {}
};

}  // namespace in_memory
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_METRIC_SPACE_DATAPOINT_H_