#include "include/parcluster/api/in-memory-metric-clusterer-base.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace research_graph {
namespace in_memory {

absl::Status InMemoryMetricClusterer::DataPoints::StartImport() {
  return absl::OkStatus();
}

absl::Status InMemoryMetricClusterer::DataPoints::AddPoint(DataPoint point) {
  return absl::OkStatus();
}

}  // namespace in_memory
}  // namespace research_graph
