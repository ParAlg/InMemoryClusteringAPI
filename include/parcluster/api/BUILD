package(default_visibility = ["//visibility:public"])

proto_library(
    name = "config_proto",
    srcs = [
        "config.proto",
    ],
    deps = ["@com_google_protobuf//:any_proto"],
)

cc_proto_library(
    name = "config_cc_proto",
    deps = [":config_proto"],
)

cc_library(
    name = "in-memory-clusterer-base",
    srcs = ["in-memory-clusterer-base.cc"],
    hdrs = ["in-memory-clusterer-base.h"],
    deps = [
        ":config_cc_proto",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@gbbs//gbbs",
    ],
)

cc_library(
    name = "in-memory-metric-clusterer-base",
    srcs = ["in-memory-metric-clusterer-base.cc"],
    hdrs = ["in-memory-metric-clusterer-base.h"],
    deps = [
        ":config_cc_proto",
        ":datapoint",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@gbbs//gbbs",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
    deps = [
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "gbbs-graph",
    srcs = ["gbbs-graph.cc"],
    hdrs = ["gbbs-graph.h"],
    deps = [
        ":in-memory-clusterer-base",
        ":status_macros",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@gbbs//gbbs",
        "@gbbs//gbbs:graph",
        "@gbbs//gbbs:macros",
    ],
)

cc_library(
    name = "datapoint",
    hdrs = ["datapoint.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@gbbs//gbbs",
    ],
)