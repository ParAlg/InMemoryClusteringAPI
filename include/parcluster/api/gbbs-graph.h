#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_GBBS_GRAPH_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_GBBS_GRAPH_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "in-memory-clusterer-base.h"
#include "gbbs/gbbs.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"

namespace research_graph {
namespace in_memory {

// Represents a weighted undirected graph in GBBS format.
// Multiple edges and self-loops are allowed.
// Note that GBBS doesn't support node weights.
// Also, Import does not automatically symmetrize the graph. If a vertex u is in
// the adjacency list of a vertex v, then it is not guaranteed that vertex v
// will appear in the adjacency list of vertex u unless explicitly
// specified in vertex u's adjacency list.
class GbbsGraph : public InMemoryClusterer::Graph {
 public:
  // Stores the node and edge information in nodes_ and edges_
  absl::Status Import(AdjacencyList adjacency_list) override;
  absl::Status PrepareImport(int64_t num_nodes) override;
  // Constructs graph_ using nodes_ and edges_
  absl::Status FinishImport() override;
  NodeId Degree(NodeId i) const override {return graph_->get_vertex(i).out_degree();}

  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* Graph() const;

 private:
  // Ensures that the graph has the given number of nodes, by adding new nodes
  // if necessary.
  void EnsureSize(NodeId id);
  absl::Mutex mutex_;
  std::vector<gbbs::symmetric_vertex<float>> nodes_;
  std::vector<std::unique_ptr<std::tuple<gbbs::uintE, float>[]>> edges_;
  std::shared_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      graph_;
  int64_t num_nodes_ = 0;
};

// Calls out_graph->Import() for each node in in_graph.
absl::Status CopyGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& in_graph,
    InMemoryClusterer::Graph* out_graph);

}  // namespace in_memory
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_GBBS_GRAPH_H_
