#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_METRIC_SPACE_IN_MEMORY_CLUSTERER_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_METRIC_SPACE_IN_MEMORY_CLUSTERER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "include/parcluster/api/config.pb.h"
#include "include/parcluster/api/datapoint.h"
#include "external/gbbs/gbbs/gbbs.h"

namespace research_graph {
namespace in_memory {

// Interface of an in-memory metric clustering algorithm. The classes
// implementing this interface maintain a mutable set of datapoints, which can
// be clustered using a given set of parameters.
class InMemoryMetricClusterer {
 public:
  virtual ~InMemoryMetricClusterer() {}

  // Clusters the currently maintained set of datapoints using the given set of
  // parameters.
  // Returns a clustering, or an error if the algorithm failed to cluster the
  // given datapoints.
  // Note that the same clustering may have multiple representations, and the
  // function may return any of them.
  virtual absl::StatusOr<std::vector<int64_t>> Cluster(
      absl::Span<DataPoint> datapoints,
      const MetricClustererConfig& config) const = 0;
};

}  // namespace in_memory
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_METRIC_SPACE_IN_MEMORY_CLUSTERER_H_
