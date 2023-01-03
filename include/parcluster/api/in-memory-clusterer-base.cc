#include "parcluster/api/in-memory-clusterer-base.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace research_graph {
namespace in_memory {

absl::Status InMemoryClusterer::Graph::FinishImport() {
  return absl::OkStatus();
}

std::string InMemoryClusterer::StringId(NodeId id) const {
  if (node_id_map_ == nullptr) {
    return absl::StrCat(id);
  } else if (id < node_id_map_->size()) {
    return (*node_id_map_)[id];
  } else {
    return absl::StrCat("missing-id-", id);
  }
}

absl::StatusOr<InMemoryClusterer::Dendrogram>
InMemoryClusterer::HierarchicalCluster (
    const ClustererConfig& config) const {
  return absl::UnimplementedError("HierarchicalCluster not implemented.");
}

}  // namespace in_memory
}  // namespace research_graph
