#include "parallel-graph-utils.h"

#include <cstdio>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "gbbs/gbbs.h"
#include "gbbs/graph_io.h"
#include "gbbs/macros.h"
#include "parallel-sequence-ops.h"

namespace research_graph {

std::vector<gbbs::uintE> GetOffsets(
    const std::function<gbbs::uintE(std::size_t)>& get_key,
    gbbs::uintE num_keys, std::size_t n) {
  std::vector<gbbs::uintE> offsets(n + 1, 0);
  // Obtain the boundary indices where keys differ
  // These indices are stored in filtered_mark_keys
  std::vector<gbbs::uintE> filtered_mark_keys =
      parallel::GetBoundaryIndices<gbbs::uintE>(
          num_keys, [&get_key](std::size_t i, std::size_t j) {
            return get_key(i) == get_key(j);
          });
  std::size_t num_filtered_mark_keys = filtered_mark_keys.size() - 1;

  // We must do an extra step for keys i which do not appear in get_key
  // At the start of each boundary index start_index, the first key
  // is given by get_key(start_index). The offset for that key is precisely
  // start_index. The offset for each key after get_key(start_index - 1) to
  // get_key(start_index) is also start_index, because these keys do
  // not appear in get_key.
  parlay::parallel_for(0, num_filtered_mark_keys + 1, [&](std::size_t i) {
    auto start_index = filtered_mark_keys[i];
    gbbs::uintE curr_key = start_index == num_keys ? n : get_key(start_index);
    gbbs::uintE prev_key = start_index == 0 ? 0 : get_key(start_index - 1) + 1;
    for (std::size_t j = prev_key; j <= curr_key; j++) {
      offsets[j] = start_index;
    }
  });

  return offsets;
}

namespace {

// Retrieves a list of inter-cluster edges, given a set of cluster_ids
// that form the vertices of a new graph. Maps all edges in original_graph
// to cluster ids. Depending on is_valid_func, combines edges in the same
// cluster to be a self-loop.
std::vector<std::tuple<gbbs::uintE, gbbs::uintE, float>>
RetrieveInterClusterEdges(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::function<bool(gbbs::uintE, gbbs::uintE)>& is_valid_func,
    const std::function<float(std::tuple<gbbs::uintE, gbbs::uintE, float>)>& scale_func) {
  // First, compute offsets on the original graph
  std::vector<gbbs::uintE> all_offsets(original_graph.n + 1, gbbs::uintE{0});
  parlay::parallel_for(0, original_graph.n, [&](std::size_t i) {
    all_offsets[i] = original_graph.get_vertex(i).out_degree();
  });
  std::pair<parlay::sequence<gbbs::uintE>, gbbs::uintE> all_offsets_scan =
      research_graph::parallel::ScanAdd(absl::Span<const gbbs::uintE>(
          all_offsets.data(), all_offsets.size()));

  // Retrieve all edges in the graph, mapped to cluster_ids
  std::vector<std::tuple<gbbs::uintE, gbbs::uintE, float>> all_edges(
      original_graph.m, std::make_tuple(UINT_E_MAX, UINT_E_MAX, float{0}));

  parlay::parallel_for(0, original_graph.n, [&](std::size_t j) {
    auto vtx = original_graph.get_vertex(j);
    gbbs::uintE i = 0;
    if (cluster_ids[j] != UINT_E_MAX) {
      auto map_f = [&](gbbs::uintE u, gbbs::uintE v, float weight) {
        if (is_valid_func(cluster_ids[v], cluster_ids[u]) &&
            cluster_ids[v] != UINT_E_MAX
            // TODO: they do this line, which doesn't double count self loops
            // but somehow if we do double count self loops, it makes us match
            // their modularity better?
            // ah our volume of a supernode is off depending on this --
            // volume is taken to be the sum, which does double count self loops
            // so if we want to match PLM, we have to subtract weight of
            // self loop from weight of supernode, and put this line back in.
            && (v <= u || cluster_ids[v] != cluster_ids[u])
            )
          all_edges[all_offsets_scan.first[j] + i] =
              std::make_tuple(cluster_ids[u], cluster_ids[v], scale_func(std::make_tuple(u, v, weight)));
        i++;
      };
      vtx.out_neighbors().map(map_f, false);
    }
  });

  // Filter for valid edges
  std::vector<std::tuple<gbbs::uintE, gbbs::uintE, float>> filtered_edges =
      research_graph::parallel::FilterOut<
          std::tuple<gbbs::uintE, gbbs::uintE, float>>(
          absl::Span<const std::tuple<gbbs::uintE, gbbs::uintE, float>>(
              all_edges.data(), original_graph.m),
          [](std::tuple<gbbs::uintE, gbbs::uintE, float> x) {
            return std::get<0>(x) != UINT_E_MAX && std::get<1>(x) != UINT_E_MAX;
          });
  return filtered_edges;
}

}  // namespace

std::tuple<std::vector<double>, double, std::size_t> SeqComputeModularityConfig(
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph, double resolution){
  std::size_t total_edge_weight = 0;
  std::vector<double> node_weights(graph->n);
  for (std::size_t i = 0; i < graph->n; i++) {
    auto vtx = graph->get_vertex(i);
    auto wgh = vtx.out_degree();
    // TODO: this assumes unit edge weights
    total_edge_weight += wgh;
    node_weights[i] = wgh;
  }
  double new_resolution = resolution / total_edge_weight;
  return std::make_tuple(node_weights, new_resolution, total_edge_weight);
}

std::tuple<std::vector<double>, double, std::size_t> ComputeModularityConfig(
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph, double resolution){
  std::vector<double> node_weights(graph->n);
  parlay::parallel_for(0, graph->n, [&](std::size_t i){
    auto vtx = graph->get_vertex(i);
    auto wgh = vtx.out_degree();
    // TODO: this assumes unit edge weights
    node_weights[i] = wgh;
  });
  double total_edge_weight = parallel::ReduceAdd(absl::Span<const double>(node_weights));
  double new_resolution = resolution / total_edge_weight;
  return std::make_tuple(node_weights, new_resolution, total_edge_weight);
}


OffsetsEdges ComputeInterClusterEdgesSort(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    std::size_t num_compressed_vertices,
    const std::function<float(float, float)>& aggregate_func,
    const std::function<bool(gbbs::uintE, gbbs::uintE)>& is_valid_func,
    const std::function<float(std::tuple<gbbs::uintE, gbbs::uintE, float>)>& scale_func) {
  // Retrieve all valid edges, mapped to cluster_ids
  auto inter_cluster_edges =
      RetrieveInterClusterEdges(original_graph, cluster_ids, is_valid_func, scale_func);

  // Sort inter-cluster edges and obtain boundary indices where edges differ
  // (in any vertex). These indices are stored in filtered_mark_edges.
  auto get_endpoints =
      [](const std::tuple<gbbs::uintE, gbbs::uintE, float>& edge_with_weight) {
        return std::tie(std::get<0>(edge_with_weight),
                        std::get<1>(edge_with_weight));
      };
  auto inter_cluster_edges_sort = research_graph::parallel::ParallelSampleSort<
      std::tuple<gbbs::uintE, gbbs::uintE, float>>(
      absl::Span<std::tuple<gbbs::uintE, gbbs::uintE, float>>(
          inter_cluster_edges.data(), inter_cluster_edges.size()),
      [&](std::tuple<gbbs::uintE, gbbs::uintE, float> a,
          std::tuple<gbbs::uintE, gbbs::uintE, float> b) {
        return get_endpoints(a) < get_endpoints(b);
      });

  std::vector<gbbs::uintE> filtered_mark_edges =
      research_graph::parallel::GetBoundaryIndices<gbbs::uintE>(
          inter_cluster_edges_sort.size(),
          [&inter_cluster_edges_sort, &get_endpoints](std::size_t i,
                                                      std::size_t j) {
            return get_endpoints(inter_cluster_edges_sort[i]) ==
                   get_endpoints(inter_cluster_edges_sort[j]);
          });
  std::size_t num_filtered_mark_edges = filtered_mark_edges.size() - 1;

  // Filter out unique edges into edges
  // This is done by iterating over the boundaries where edges differ,
  // and retrieving all of the same edges in one section. These edges
  // are combined using aggregate_func.
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges(
      new std::tuple<gbbs::uintE, float>[num_filtered_mark_edges]);
  // Separately save the first vertex in the corresponding edges, to compute
  // offsets
  std::vector<gbbs::uintE> edges_for_offsets(num_filtered_mark_edges);
  parlay::parallel_for(0, num_filtered_mark_edges, [&](std::size_t i) {
    // Combine edges from start_edge_index to end_edge_index
    gbbs::uintE start_edge_index = filtered_mark_edges[i];
    gbbs::uintE end_edge_index = filtered_mark_edges[i + 1];
    /*float weight = 0;
    for (std::size_t k = start_edge_index; k < end_edge_index; k++) {
      weight += std::get<2>(inter_cluster_edges_sort[k]);
    }*/
    float weight = std::get<2>(research_graph::parallel::Reduce<
                               std::tuple<gbbs::uintE, gbbs::uintE, float>>(
        absl::Span<const std::tuple<gbbs::uintE, gbbs::uintE, float>>(
            inter_cluster_edges_sort.begin() + start_edge_index,
            end_edge_index - start_edge_index),
        [&](std::tuple<gbbs::uintE, gbbs::uintE, float> a,
            std::tuple<gbbs::uintE, gbbs::uintE, float> b) {
          return std::make_tuple(
              gbbs::uintE{0}, gbbs::uintE{0},
              aggregate_func(std::get<2>(a), std::get<2>(b)));
        },
        std::make_tuple(gbbs::uintE{0}, gbbs::uintE{0}, float{0})));
    edges[i] = std::make_tuple(
        std::get<1>(inter_cluster_edges_sort[start_edge_index]), weight);
    edges_for_offsets[i] =
        std::get<0>(inter_cluster_edges_sort[start_edge_index]);
  });

  // Compute offsets using filtered edges.
  auto offsets = GetOffsets(
      [&edges_for_offsets](std::size_t i) -> gbbs::uintE {
        return edges_for_offsets[i];
      },
      num_filtered_mark_edges, num_compressed_vertices);
  return OffsetsEdges{offsets, std::move(edges), num_filtered_mark_edges};
}

std::vector<gbbs::uintE> FlattenClustering(
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::vector<gbbs::uintE>& compressed_cluster_ids) {
  std::vector<gbbs::uintE> new_cluster_ids(cluster_ids.size());
  parlay::parallel_for(0, cluster_ids.size(), [&](std::size_t i) {
    new_cluster_ids[i] = (cluster_ids[i] == UINT_E_MAX)
                             ? UINT_E_MAX
                             : compressed_cluster_ids[cluster_ids[i]];
  });
  return new_cluster_ids;
}

}  // namespace research_graph
