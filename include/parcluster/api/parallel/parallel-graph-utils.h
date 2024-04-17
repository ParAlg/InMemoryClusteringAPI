#ifndef RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_
#define RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_

#include <cstdio>

#include "gbbs/gbbs.h"
#include "gbbs/graph_io.h"
#include "gbbs/macros.h"

#include "parallel-sequence-ops.h"

namespace research_graph {

struct OffsetsEdges {
  std::vector<gbbs::uintE> offsets;
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges;
  std::size_t num_edges;
};

// Given get_key, which is nondecreasing, defined for 0, ..., num_keys-1, and
// returns an unsigned integer less than n, return an array of length n + 1
// where array[i] := minimum index k such that get_key(k) >= i.
// Note that array[n] = the total number of keys, num_keys.
std::vector<gbbs::uintE> GetOffsets(
    const std::function<gbbs::uintE(std::size_t)>& get_key,
    gbbs::uintE num_keys, std::size_t n);

std::tuple<std::vector<double>, double, std::size_t> ComputeModularityConfig(
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph, double resolution);

std::tuple<std::vector<double>, double, std::size_t> SeqComputeModularityConfig(
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph, double resolution);

// Using parallel sorting, compute inter cluster edges given a set of
// cluster_ids that form the vertices of the new graph. Uses aggregate_func
// to combine multiple edges on the same cluster ids. Returns sorted
// edges and offsets array in edges and offsets respectively.
// The number of compressed vertices should be 1 + the maximum cluster id
// in cluster_ids.
OffsetsEdges ComputeInterClusterEdgesSort(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    std::size_t num_compressed_vertices,
    const std::function<float(float, float)>& aggregate_func,
    const std::function<bool(gbbs::uintE, gbbs::uintE)>& is_valid_func,
    const std::function<float(std::tuple<gbbs::uintE, gbbs::uintE, float>)>& scale_func);

// Given an array of edges (given by a tuple consisting of the second endpoint
// and a weight if the edges are weighted) and the offsets marking the index
// of the first edge corresponding to each vertex (essentially, CSR format),
// return the corresponding graph in GBBS format.
// Note that the returned graph takes ownership of the edges array.
template <typename WeightType>
std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>
MakeGbbsGraph(
    const std::vector<gbbs::uintE>& offsets, std::size_t num_vertices,
    std::unique_ptr<std::tuple<gbbs::uintE, WeightType>[]> edges_pointer,
    std::size_t num_edges) {
  gbbs::symmetric_vertex<WeightType>* vertices =
      new gbbs::symmetric_vertex<WeightType>[num_vertices];
  auto edges = edges_pointer.release();

  parlay::parallel_for(0, num_vertices, [&](std::size_t i) {
    gbbs::vertex_data vertex_data{offsets[i], offsets[i + 1] - offsets[i]};
    vertices[i] = gbbs::symmetric_vertex<WeightType>(edges, vertex_data, i);
  });

  return std::make_unique<
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>(
      num_vertices, num_edges, vertices, [=]() {
        delete[] vertices;
        delete[] edges;
      });
}

template <typename WeightType>
std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>
SeqMakeGbbsGraph(
    const std::vector<gbbs::uintE>& offsets, std::size_t num_vertices,
    std::unique_ptr<std::tuple<gbbs::uintE, WeightType>[]> edges_pointer,
    std::size_t num_edges) {
  gbbs::symmetric_vertex<WeightType>* vertices =
      new gbbs::symmetric_vertex<WeightType>[num_vertices];
  auto edges = edges_pointer.release();

  for (std::size_t i = 0; i < num_vertices; i++) {
    gbbs::vertex_data vertex_data{offsets[i], offsets[i + 1] - offsets[i]};
    vertices[i] = gbbs::symmetric_vertex<WeightType>(edges, vertex_data, i);
  }

  return std::make_unique<
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>(
      num_vertices, num_edges, vertices, [=]() {
        delete[] vertices;
        delete[] edges;
      });
}

// Given new cluster ids in compressed_cluster_ids, remap the original
// cluster ids. A cluster id of UINT_E_MAX indicates that the vertex
// has already been placed into a finalized cluster, and this is
// preserved in the remapping.
std::vector<gbbs::uintE> FlattenClustering(
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::vector<gbbs::uintE>& compressed_cluster_ids);

// Map DenseClustering type to nested vector type. Each vector in nested vector is a community.
template <class NodeId, class DenseClustering>
inline std::vector<std::vector<NodeId>> DenseClusteringToNestedClustering(
    const DenseClustering& clustering) {
  auto pairs = parlay::sequence<std::pair<NodeId, NodeId>>::from_function(clustering.size(), [&] (NodeId i) {
    return std::make_pair(clustering[i], i);
  });
  // parlay::sort_inplace(parlay::make_slice(pairs));
  // for (long i=0; i<pairs.size(); i++) {
  //   auto& cluster_i = pairs[i].first;
  //   if (i == 0 || cluster_i != pairs[i-1].first) {
  //     output.emplace_back(std::vector<NodeId>{pairs[i].second});
  //   } else {
  //     output[output.size()-1].emplace_back(pairs[i].second);
  //   }
  // }
  auto grouped = parlay::group_by_key(pairs);
  std::vector<std::vector<NodeId>> output(grouped.size());
  parlay::parallel_for(0, grouped.size(), [&](size_t i){
    output[i].resize(grouped[i].second.size());
    parlay::parallel_for(0, grouped[i].second.size(), [&](size_t j){
      output[i][j] = grouped[i].second[j];
    });
    // output[i] = std::vector<NodeId>(grouped[i].second.begin(), grouped[i].second.end());
  });

  return output;
}


// Holds a GBBS graph and a corresponding node weights
template <typename NodeWeightType>
struct GraphWithWeights {
  GraphWithWeights() {}
  GraphWithWeights(
      std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
          graph_,
      std::vector<NodeWeightType> node_weights_)
      : graph(std::move(graph_)), node_weights(std::move(node_weights_)) {}
  std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      graph;
  std::vector<NodeWeightType> node_weights;
};

}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_
