# Vector Index

`ggml-vector-index` is an opt-in C API for local vector search. It stores caller
provided ids with dense vectors and exposes exact top-k search over f32, q8, and
packed q4 storage.

This candidate component is currently standalone. It is not enabled in default
builds and is not wired into the llama runtime, server, or app paths. Consumers
should enable it explicitly and link the vector-index target directly.

## How The Pieces Fit

The public API owns an opaque index handle, caller ids, storage slots, and the
exact top-k baseline used to check later search modes. Storage starts with f32
for correctness, then q8 and packed q4 trade precision for smaller CPU-resident
rows while keeping the query path in f32.

Search layers build upward from exact scan: filtered search limits candidates by
id, prepared filters reuse that id-to-slot mapping, and IVF-flat adds an
in-memory candidate selector. Persistence is layered separately: `.tvim` full
snapshots can be loaded normally or mmap-backed for read-only search, and
`.tvid` logs replay or compact mutations on top of a snapshot. TurboVec then
adds the q2/q4 path: rotation, TQ+ calibration, codebooks, bit-plane storage,
LUT scoring, and blocked SIMD cache state. Tests cover each layer where it
lands, with fault and cross-process checks reserved for durable persistence.

## Build

Enable the library with `GGML_VECTOR_INDEX`:

```sh
cmake -B build -DGGML_VECTOR_INDEX=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build --target ggml-vector-index test-vector-index
```

Installed CMake packages export the target as `ggml::vector-index`.

```cmake
target_link_libraries(my_target PRIVATE ggml::vector-index)
```

NEON kernels are selected automatically on supported ARM builds. On x86,
AVX2 kernels are built separately when `GGML_NATIVE=ON` or `GGML_AVX2=ON` and
are selected at runtime. Scalar and SIMD reduction order can produce small
score differences across CPU architectures.

## Storage Modes

Create an index with a fixed dimension and bit width:

- `bit_width=32`: full f32 vectors.
- `bit_width=8`: per-vector symmetric q8 storage with f32 scales.
- `bit_width=4`: per-vector symmetric packed q4 storage with f32 scales.

Search scores are dot products. The index does not normalize vectors internally.
For cosine similarity, normalize vectors before insertion and normalize queries
before search.

## Search Modes

Exact search scans all live slots. `ggml_vec_index_search_filtered` restricts
that scan to caller-provided ids, and `ggml_vec_index_filter_create` prepares
the same id set for repeated `ggml_vec_index_search_prepared_filtered` calls.

`ggml_vec_index_build_ivf` builds heap-owned IVF-flat state for approximate
candidate selection. Call it again after loading an index and after successful
add/remove mutations. IVF state is not persisted in snapshots.

## Persistence

Snapshots use `.tvim`. Version 2 records the storage kind, ids, quantization
scales, vector bytes, and checksums. The loader still accepts legacy v1 f32
snapshots; legacy `bit_width=8` files are quantized to q8 on load.

`ggml_vec_index_load_mmap` maps the vector section read-only and copies ids and
scales into memory. mmap-loaded handles allow search and IVF preparation, but
reject content mutations.
