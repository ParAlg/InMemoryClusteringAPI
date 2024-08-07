#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_IN_MEMORY_CLUSTERER_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_IN_MEMORY_CLUSTERER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "parcluster/api/config.pb.h"
#include "gbbs/gbbs.h"

namespace research_graph {
namespace in_memory {

// Interface of an in-memory clustering algorithm. The classes implementing this
// interface maintain a mutable graph, which can be clustered using a given set
// of parameters.
class InMemoryClusterer {
 public:
  // This is a basic interface for building graphs. Note that the interface only
  // specifies how to build a graph, as different clusterers may use different
  // interfaces for accessing it.
  // The node ids are consecutive, 0-based integers. In particular, adding a
  // node of id k to an empty graph creates k+1 nodes 0, ..., k.
  class Graph {
   public:
    using NodeId = gbbs::uintE;

    // Represents a weighted node with weighted outgoing edges.
    struct AdjacencyList {
      NodeId id = -1;
      double weight = 1;
      std::vector<std::pair<NodeId, double>> outgoing_edges;
    };

    virtual ~Graph() = default;

    // Adds a weighted node and its weighted out-edges to the graph. Depending
    // on the Graph implementation, the symmetric edges may be added as well,
    // and edge weights may be adjusted for symmetry.
    //
    // Import must be called at most once for each node. If not called for a
    // node, that node defaults to weight 1.
    //
    // IMPLEMENTATIONS MUST ALLOW CONCURRENT CALLS TO Import()!
    virtual absl::Status Import(AdjacencyList adjacency_list) = 0;
    virtual absl::Status PrepareImport(int64_t num_nodes) = 0;
    virtual absl::Status FinishImport();
    virtual NodeId Degree(NodeId i) const = 0;
  };

  using NodeId = Graph::NodeId;
  using AdjacencyList = Graph::AdjacencyList;

  // TODO(jeshi): This is a temporary dendrogram object that only stores the
  // parent (and not associated data such as similarity). It should be
  // replaced with the internal dendrogram object.
  using Dendrogram = std::vector<gbbs::uintE>;

  // Represents clustering: each element of the vector contains the set of
  // NodeIds in one cluster. We call a clustering non-overlapping if the
  // elements of the clustering are nonempty vectors that together contain each
  // NodeId exactly once.
  using Clustering = std::vector<std::vector<NodeId>>;

  virtual ~InMemoryClusterer() {}

  // Accessor to the maintained graph. Use it to build the graph.
  virtual Graph* MutableGraph() = 0;

  // Clusters the currently maintained graph using the given set of parameters.
  // Returns a clustering, or an error if the algorithm failed to cluster the
  // given graph.
  // Note that the same clustering may have multiple representations, and the
  // function may return any of them.
  virtual absl::StatusOr<Clustering> Cluster(
      const ClustererConfig& config) const = 0;
  
  // Returns a family of clusterings represented by a dendrogram in the parent-
  // array format. Note that the default implementation returns an error status,
  // so callers should ensure that the Clusterer being used implements this
  // method.
  virtual absl::StatusOr<Dendrogram> HierarchicalCluster(
      const ClustererConfig& config) const;

  // Refines a list of clusters and redirects the given pointer to new clusters.
  // This function is useful for methods that can refine / operate on an
  // existing clustering. It does not take ownership of clustering. The default
  // implementation does nothing and returns OkStatus.
  virtual absl::Status RefineClusters(const ClustererConfig& config,
                                      Clustering* clustering) const {
    return absl::OkStatus();
  }

  // Provides a pointer to a vector that contains string ids corresponding to
  // the NodeIds. If set, the ids from the provided map are used in the log and
  // error messages. The vector must live during all method calls of this
  // object. This call does *not* take ownership of the pointee. Using this
  // function is not required. If this function is never called, the ids are
  // converted to strings using absl::StrCat.
  void set_node_id_map(const std::vector<std::string>* node_id_map) {
    node_id_map_ = node_id_map;
  }

 protected:
  // Returns the string id corresponding to a given NodeId. If set_node_id_map
  // was called, uses the map to get the ids. Otherwise, returns the string
  // representation of the id.
  std::string StringId(NodeId id) const;

 private:
  // NodeId map set by set_node_id_map(). May be left to nullptr even after
  // initialization.
  const std::vector<std::string>* node_id_map_ = nullptr;
};

}  // namespace in_memory
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_IN_MEMORY_CLUSTERER_H_
