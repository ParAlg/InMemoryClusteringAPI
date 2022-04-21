#include "parcluster/api/gbbs-graph.h"

#include <algorithm>
#include <memory>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "status_macros.h"
#include "gbbs/macros.h"

namespace research_graph {
namespace in_memory {

void GbbsGraph::EnsureSize(NodeId id) {
  if (nodes_.size() < id) nodes_.resize(id, gbbs::symmetric_vertex<float>());
}

// TODO(jeshi,laxmand): should adjacency_list be a const&?
absl::Status GbbsGraph::Import(AdjacencyList adjacency_list) {
  using GbbsEdge = std::tuple<gbbs::uintE, float>;
  // TODO(jeshi): Parallelize further by only taking the mutex when
  // nodes_ and edges_ need to double in size, instead of every time
  // a node id is increased.
  auto outgoing_edges_size = adjacency_list.outgoing_edges.size();
  auto out_neighbors = absl::make_unique<GbbsEdge[]>(outgoing_edges_size);
  gbbs::parallel_for(0, outgoing_edges_size, [&](size_t i) {
    out_neighbors[i] = std::make_tuple(
        static_cast<gbbs::uintE>(adjacency_list.outgoing_edges[i].first),
        adjacency_list.outgoing_edges[i].second);
  });
  absl::MutexLock lock(&mutex_);
  EnsureSize(adjacency_list.id + 1);
  nodes_[adjacency_list.id].degree = outgoing_edges_size;
  nodes_[adjacency_list.id].neighbors = out_neighbors.get();
  nodes_[adjacency_list.id].id = adjacency_list.id];
  if (edges_.size() <= adjacency_list.id) edges_.resize(adjacency_list.id + 1);
  edges_[adjacency_list.id] = std::move(out_neighbors);

  return absl::OkStatus();
}

absl::Status GbbsGraph::FinishImport() {
  auto degrees = parlay::sequence<gbbs::uintE>::from_function(
      nodes_.size(), [this](size_t i) { return nodes_[i].out_degree(); });
  auto num_edges = parlay::reduce(parlay::make_slice(degrees));

  auto neighbors = parlay::sequence<gbbs::uintE>::from_function(nodes_.size(), [this](size_t i) {
    if (nodes_[i].out_degree() == 0) return gbbs::uintE{0};
    auto map_f = [&] (const auto& u, const auto& v, const auto& wgh) { return v; };
    auto max_neighbor = nodes_[i].out_neighbors().reduce(map_f, parlay::maxm<gbbs::uintE>());
    return max_neighbor;
  });
  auto max_node = parlay::reduce_max(parlay::make_slice(neighbors));
  EnsureSize(max_node + 1);

  // The GBBS graph takes no ownership of nodes / edges
  auto g = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      nodes_.size(), num_edges, nodes_.data(), []() {});  // noop deletion_fn
  graph_ = std::make_shared<
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>(g);
  return absl::OkStatus();
}

// TODO: What about compressed graphs?
gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* GbbsGraph::Graph()
    const {
  return graph_.get();
}

// TODO: What about compressed graphs?
absl::Status CopyGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& in_graph,
    InMemoryClusterer::Graph* out_graph) {
  for (gbbs::uintE id = 0; id < in_graph.n; id++) {
    InMemoryClusterer::AdjacencyList adjacency_list;
    adjacency_list.id = id;
    //using edge_type = typename gbbs::symmetric_vertex<float>::edge_type;
    //const auto& neighbors = (edge_type*)in_graph.get_vertex(id).out_neighbors();
    adjacency_list.outgoing_edges.reserve(
        in_graph.get_vertex(id).out_degree());
    auto map_f = [&] (const auto& u, const auto& v, const auto& wgh) {
      adjacency_list.outgoing_edges.emplace_back(v, wgh);
    };
    in_graph.get_vertex(id).out_neighbors().map(map_f, /* parallel = */ false);
    RETURN_IF_ERROR(out_graph->Import(std::move(adjacency_list)));
  }
  RETURN_IF_ERROR(out_graph->FinishImport());
  return absl::OkStatus();
}

}  // namespace in_memory
}  // namespace research_graph
