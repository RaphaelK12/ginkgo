constexpr auto spgemm_block_size = 128;
constexpr auto spgemm_child_count = 2;

template <int max_num_rows, bool use_val, typename ValueType,
          typename IndexType, typename StepCallback, typename ColCallback>
__device__ void spgemm_multiway_merge_short(
    IndexType a_size, const IndexType *__restrict__ a_cols,
    const ValueType *__restrict__ a_vals,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    StepCallback step_cb, ColCallback col_cb)
{
    // short rows of a: we can keep the state for each row of b in registers
    // partitioning of data: thread groups of size subsubwarp_size store the
    // content of a single row of b as a shift register. We only need to look
    // at the first element in each group, but can avoid many smaller loads.
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    constexpr auto warp_size = config::warp_size;
    constexpr auto subwarp_size = warp_size / max_num_rows;
    auto warp = group::tiled_partition<warp_size>(group::this_thread_block());
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());

    const auto warp_rank = warp.thread_rank();
    const auto subwarp_rank = subwarp.thread_rank();
    const auto subwarp_id = static_cast<IndexType>(warp_rank / subwarp_size);
    const auto subwarp_base = subwarp_id * subwarp_size;
    // load the first subsubwarp elements for each row of b
    const auto a_idx = subwarp_id;
    const auto idle_subwarp = a_idx >= a_size;
    const auto a_col = checked_load(a_cols, a_idx, a_size, sentinel);
    const auto a_val =
        (use_val && a_idx < a_size) ? a_vals[a_idx] : zero<ValueType>();
    const auto b_begin = idle_subwarp ? IndexType{} : b_row_ptrs[a_col];
    const auto b_end = idle_subwarp ? IndexType{} : b_row_ptrs[a_col + 1];
    auto b_idx = b_begin + static_cast<IndexType>(subwarp_rank);
    auto b_col = checked_load(b_cols, b_idx, b_end, sentinel);
    auto b_val = (use_val && b_idx < b_end) ? b_vals[b_idx] : zero<ValueType>();
    auto subwarp_left = subwarp_size;
    do {
        // compute the minimum column
        auto min_col = b_col;
#pragma unroll
        for (auto i = subwarp_size; i < warp_size; i *= 2) {
            min_col = min(min_col, warp.shfl_xor(min_col, i));
        }
        // min_col is only correct for the first thread in the subwarp!
        auto min_mask = warp.ballot(b_col == min_col);
        auto sub_is_min = bool((min_mask >> subwarp_base) & 1);
        // accumulate all products
        auto cur_val = sub_is_min ? a_val * b_val : zero<ValueType>();
        // TODO does this really improve the runtime?
        if (popcnt(min_mask) == 1) {
            cur_val = warp.shfl(cur_val, ffs(min_mask) - 1);
        } else {
#pragma unroll
            for (auto i = subwarp_size; i < warp_size; i *= 2) {
                cur_val += warp.shfl_xor(cur_val, i);
            }
        }
        // output
        step_cb(cur_val, min_col);
        col_cb(min_col);
        // advance elements
        if (sub_is_min) {
            b_idx++;
            if (subwarp_size == 1) {
                b_col = checked_load(b_cols, b_idx, b_end, sentinel);
                b_val = (use_val && b_idx < b_end) ? b_vals[b_idx]
                                                   : zero<ValueType>();
            } else {
                subwarp_left--;
                b_col = subwarp.shfl_down(b_col, 1);
                b_val = subwarp.shfl_down(b_val, 1);
                // if there are none left: load new elements
                if (subwarp_left == 0) {
                    b_col = checked_load(b_cols, b_idx, b_end, sentinel);
                    b_val = (use_val && b_idx < b_end) ? b_vals[b_idx]
                                                       : zero<ValueType>();
                    subwarp_left = subwarp_size;
                }
            }
        }
    } while (warp.any(b_col < sentinel));
}


template <typename T>
__forceinline__ __device__ void swap(T &a, T &b)
{
    auto tmp = a;
    a = b;
    b = tmp;
}


template <typename T>
__forceinline__ __device__ void replace_if(bool use_new, T &val, T &new_val,
                                           T &storage)
{
    storage = use_new ? val : new_val;
    val = use_new ? new_val : val;
}


template <typename Callback>
__device__ void forall_bits(config::lane_mask_type mask, Callback cb)
{
    while (mask) {
        cb(ffs(mask) - 1);
        mask &= mask - 1;
    }
}


template <bool use_val, typename ValueType, typename IndexType,
          typename StepCallback, typename ColCallback>
__device__ void spgemm_multiway_merge_medium(
    IndexType a_size, const IndexType *__restrict__ a_cols,
    const ValueType *__restrict__ a_vals,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    IndexType *sh_cols, IndexType *sh_idxs, IndexType *sh_ends,
    ValueType *sh_vals, StepCallback step_cb, ColCallback col_cb)
{
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    constexpr auto warp_size = config::warp_size;
    constexpr auto subwarp_size = spgemm_child_count;
    auto thread_block = group::this_thread_block();
    auto warp = group::tiled_partition<warp_size>(thread_block);
    auto subwarp = group::tiled_partition<subwarp_size>(thread_block);
    const auto warp_rank = warp.thread_rank();
    // the first threadIdx in the warp
    const auto warp_base = threadIdx.x / warp_size * warp_size;
    const auto subwarp_rank = subwarp.thread_rank();
    const auto subwarp_id = warp_rank / subwarp_size;
    // first thread from the subwarp (warp-local)
    const auto subwarp_base = subwarp_id * subwarp_size;
    // lane mask containing only the current subwarp (warp-local)
    const auto subwarp_mask = ((config::lane_mask_type{1} << subwarp_size) - 1)
                              << subwarp_base;
    // index for shared-memory heap
    auto heap_idx = [&](int thread, int child) {
        return (thread + warp_base) * subwarp_size + child;
    };
    auto min_op = [](IndexType a, IndexType b) { return min(a, b); };
    auto sum_op = [](ValueType a, ValueType b) { return a + b; };

    // build heap
    auto a_col = a_cols[warp_rank];
    auto a_val = use_val ? a_vals[warp_rank] : zero<ValueType>();
    auto b_idx = b_row_ptrs[a_col];
    auto b_end = b_row_ptrs[a_col + 1];
    auto b_col = checked_load(b_cols, b_idx, b_end, sentinel);
    for (int i = 0; i < subwarp_size; ++i) {
        // memory barrier
        warp.sync();
        auto a_idx = static_cast<IndexType>((i + 1) * warp_size + warp_rank);
        auto a_new_col = checked_load(a_cols, a_idx, a_size, sentinel);
        auto a_new_val =
            (a_idx < a_size && use_val) ? a_vals[a_idx] : zero<ValueType>();
        // ensure that skipped a_cols result in sentinel b_cols
        auto skip = a_new_col == sentinel;
        auto b_new_idx = skip ? IndexType{} : b_row_ptrs[a_new_col];
        auto b_new_end = skip ? IndexType{} : b_row_ptrs[a_new_col + 1];
        auto b_new_col = checked_load(b_cols, b_new_idx, b_new_end, sentinel);
        auto better = b_new_col < b_col;
        // store the entry with the larger column in shared memory
        auto sh_idx = heap_idx(warp_rank, i);
        replace_if(better, b_col, b_new_col, sh_cols[sh_idx]);
        replace_if(better, b_idx, b_new_idx, sh_idxs[sh_idx]);
        replace_if(better, b_end, b_new_end, sh_ends[sh_idx]);
        if (use_val) {
            replace_if(better, a_val, a_new_val, sh_vals[sh_idx]);
        }
    }

    auto min_col = reduce(warp, b_col, min_op);
    do {
        // find all minimal heap elements
        auto min_mask = warp.ballot(b_col == min_col);
        auto cur_val = (b_col == min_col && use_val) ? a_val * b_vals[b_idx]
                                                     : zero<ValueType>();
        if (popcnt(min_mask) == 1) {
            cur_val = warp.shfl(cur_val, ffs(min_mask) - 1);
        } else {
            cur_val = reduce(warp, cur_val, sum_op);
        }
        step_cb(cur_val, min_col);
        // update the heap
        if (b_col == min_col) {
            b_idx++;
            b_col = checked_load(b_cols, b_idx, b_end, sentinel);
        }
        // for all threads in subwarp: swap with child entries if necessary
        forall_bits(min_mask & subwarp_mask, [&](int thread) {
            // memory barrier
            subwarp.sync();
            auto sh_idx = heap_idx(thread, subwarp_rank);
            // find its minimum child
            auto b_new_col = sh_cols[sh_idx];
            auto local_min_col = reduce(subwarp, b_new_col, min_op);
            auto local_is_min = subwarp.ballot(local_min_col == b_new_col);
            auto local_min_thread = ffs(local_is_min) - 1;
            // and update its local data if necessary
            if (warp_rank == thread && local_min_col < b_col) {
                sh_idx = heap_idx(warp_rank, local_min_thread);
                swap(b_col, sh_cols[sh_idx]);
                swap(b_idx, sh_idxs[sh_idx]);
                swap(b_end, sh_ends[sh_idx]);
                if (use_val) {
                    swap(a_val, sh_vals[sh_idx]);
                }
            }
        });
        auto new_min_col = reduce(warp, b_col, min_op);
        if (new_min_col > min_col) {
            col_cb(min_col);
        }
        min_col = new_min_col;
    } while (min_col < sentinel);
}


template <bool use_val, typename ValueType, typename IndexType,
          typename StepCallback, typename ColCallback>
__device__ void spgemm_multiway_merge_large(
    IndexType a_size, const IndexType *__restrict__ a_cols,
    const ValueType *__restrict__ a_vals,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    IndexType *__restrict__ sh_cols, IndexType *__restrict__ sh_idxs,
    IndexType *__restrict__ sh_ends, ValueType *__restrict__ sh_vals,
    IndexType *__restrict__ gl_cols, IndexType *__restrict__ gl_idxs,
    IndexType *__restrict__ gl_ends, ValueType *__restrict__ gl_vals,
    StepCallback step_cb, ColCallback col_cb)
{
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    constexpr auto warp_size = config::warp_size;
    constexpr auto subwarp_size = spgemm_child_count;
    constexpr auto subwarp_count = warp_size / subwarp_size;
    // reduction operations
    auto min_op = [](IndexType a, IndexType b) { return min(a, b); };
    auto sum_op = [](ValueType a, ValueType b) { return a + b; };

    auto thread_block = group::this_thread_block();

    // warp cooperative group for global minimum and sum reduction
    auto warp = group::tiled_partition<warp_size>(thread_block);
    // rank of the thread within the warp (= sub-heap idx)
    const auto warp_rank = warp.thread_rank();
    // ID of the warp within the block
    const auto warp_id = threadIdx.x / warp_size;
    // first thread in the warp (block-local, for shared memory access)
    const auto warp_base = warp_id * warp_size;

    // subwarp cooperative group for heap updates
    auto subwarp = group::tiled_partition<subwarp_size>(thread_block);
    // rank of the thread within the subwarp
    const auto subwarp_rank = subwarp.thread_rank();
    // ID of the subwarp within the warp
    const auto subwarp_id = warp_rank / subwarp_size;
    // first thread from the subwarp (warp-local)
    const auto subwarp_base = subwarp_id * subwarp_size;
    // lane mask containing only the current subwarp (warp-local)
    const auto subwarp_mask = ((config::lane_mask_type{1} << subwarp_size) - 1)
                              << subwarp_base;

    // child index for shared-memory heap
    auto sh_child_idx = [&](int thread, int child) {
        return (thread + warp_base) * spgemm_child_count + child;
    };
    // translates a shared-memory heap index into a global-memory heap index
    auto sh_to_gl = [&](int idx) {
        return static_cast<IndexType>(idx - warp_base * spgemm_child_count +
                                      warp_size);
    };
    // child index for global-memory heap
    auto child_idx = [](IndexType pos, IndexType child) {
        return static_cast<IndexType>(pos * spgemm_child_count + warp_size +
                                      child);
    };
    // restore the global-memory heap property from node i downwards
    auto sift_down = [&](IndexType i) {
        auto parent_col = gl_cols[i];
        // as long as i has any children:
        while (child_idx(i, 0) < a_size) {
            // memory barrier
            subwarp.sync();
            // find minimum child
            auto child = child_idx(i, subwarp_rank);
            auto child_col = checked_load(gl_cols, child, a_size, sentinel);
            auto min_col = reduce(subwarp, child_col, min_op);
            if (min_col >= parent_col) {
                break;
            }
            // swap minimum child up
            auto min_mask = subwarp.ballot(min_col == child_col);
            auto min_child = ffs(min_mask) - 1;
            if (subwarp_rank == min_child) {
                swap(gl_cols[i], gl_cols[child]);
                swap(gl_idxs[i], gl_idxs[child]);
                swap(gl_ends[i], gl_ends[child]);
                if (use_val) {
                    swap(gl_vals[i], gl_vals[child]);
                }
            }
            // proceed from location of minimum child
            i = child_idx(i, min_child);
        }
    };

    // fill heap storage
    for (int i = warp_rank; i < a_size; i += warp_size) {
        auto a_col = a_cols[i];
        auto b_begin = b_row_ptrs[a_col];
        auto b_end = b_row_ptrs[a_col + 1];
        gl_idxs[i] = b_begin;
        gl_ends[i] = b_end;
        gl_cols[i] = checked_load(b_cols, b_begin, b_end, sentinel);
        if (use_val) {
            gl_vals[i] = a_vals[i];
        }
    }
    // build heap: each subwarp handles one node
    // this should not create overlaps/races between children/parents
    auto last_parent = (a_size - 1 - warp_size) / spgemm_child_count;
    for (IndexType i = last_parent - subwarp_id; i >= 0; i -= subwarp_count) {
        // memory barrier
        subwarp.sync();
        sift_down(i);
    }

    // memory barrier
    warp.sync();
    // load top for each sub-heap into registers
    auto b_col = gl_cols[warp_rank];
    auto b_idx = gl_idxs[warp_rank];
    auto b_end = gl_ends[warp_rank];
    auto a_val = use_val ? gl_vals[warp_rank] : zero<ValueType>();
    // load next level of heap into shared memory
    for (int i = warp_rank; i < spgemm_child_count * warp_size;
         i += warp_size) {
        auto sh_idx = i + warp_base * spgemm_child_count;
        sh_cols[sh_idx] = gl_cols[i + warp_size];
        sh_idxs[sh_idx] = gl_idxs[i + warp_size];
        sh_ends[sh_idx] = gl_ends[i + warp_size];
        if (use_val) {
            sh_vals[sh_idx] = gl_vals[i + warp_size];
        }
    }

    auto min_col = reduce(warp, b_col, min_op);
    do {
        // find all heap tops that are minimal
        auto min_mask = warp.ballot(b_col == min_col);
        // compute current product
        auto cur_val = (b_col == min_col && use_val) ? a_val * b_vals[b_idx]
                                                     : zero<ValueType>();
        // compute current partial sum and call callback
        if (popcnt(min_mask) == 1) {
            cur_val = warp.shfl(cur_val, ffs(min_mask) - 1);
        } else {
            cur_val = reduce(warp, cur_val, sum_op);
        }
        step_cb(cur_val, min_col);

        // update the heap
        if (b_col == min_col) {
            b_idx++;
            b_col = checked_load(b_cols, b_idx, b_end, sentinel);
        }
        // for all threads in subwarp: restore heap property
        forall_bits(min_mask & subwarp_mask, [&](int heap) {
            // memory barrier
            subwarp.sync();
            // i'th thread loads i'th child
            auto sh_idx = sh_child_idx(heap, subwarp_rank);
            // find its minimum child
            auto child_col = sh_cols[sh_idx];
            auto heap_top_col = subwarp.shfl(b_col, heap - subwarp_base);
            auto min_child_col = reduce(subwarp, child_col, min_op);
            auto min_child_mask = subwarp.ballot(min_child_col == child_col);
            auto min_child = ffs(min_child_mask) - 1;

            // and update its local data if necessary
            if (min_child_col < heap_top_col) {
                sh_idx = sh_child_idx(heap, min_child);
                // if this is the warp holding heap's top in registers: swap
                if (warp_rank == heap) {
                    swap(b_col, sh_cols[sh_idx]);
                    swap(b_idx, sh_idxs[sh_idx]);
                    swap(b_end, sh_ends[sh_idx]);
                    if (use_val) {
                        swap(a_val, sh_vals[sh_idx]);
                    }
                }
                // update global memory heap with shared memory heap
                auto heap_idx = sh_to_gl(sh_idx);
                if (child_idx(heap_idx, 0) < a_size) {
                    // find minimum child in global heap
                    auto cur_col =
                        checked_load(gl_cols, child_idx(heap_idx, subwarp_rank),
                                     a_size, sentinel);
                    auto min_col = reduce(subwarp, cur_col, min_op);
                    // if heap property is violated: swap min child up
                    if (min_col < sh_cols[sh_idx]) {
                        min_child_mask = subwarp.ballot(min_col == cur_col);
                        min_child = ffs(min_child_mask) - 1;
                        auto new_heap_idx = child_idx(heap_idx, min_child);
                        if (subwarp_rank == 0) {
                            swap(sh_cols[sh_idx], gl_cols[new_heap_idx]);
                            swap(sh_idxs[sh_idx], gl_idxs[new_heap_idx]);
                            swap(sh_ends[sh_idx], gl_ends[new_heap_idx]);
                            if (use_val) {
                                swap(sh_vals[sh_idx], gl_vals[new_heap_idx]);
                            }
                        }
                        // memory barrier
                        subwarp.sync();
                        sift_down(new_heap_idx);
                    }
                }
            }
        });

        // call finish row callback if we are done
        auto new_min_col = reduce(warp, b_col, min_op);
        if (new_min_col > min_col) {
            col_cb(min_col);
        }
        min_col = new_min_col;
    } while (min_col < sentinel);
}


template <bool use_val, typename ValueType, typename IndexType,
          typename StepCallback, typename ColCallback>
__device__ void spgemm_multiway_merge_dispatch(
    IndexType a_size, const IndexType *__restrict__ a_cols,
    const ValueType *__restrict__ a_vals,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    IndexType *__restrict__ tmp, ValueType *__restrict__ tmpval,
    StepCallback step_cb, ColCallback col_cb)
{
    constexpr auto heap_size = config::warp_size * spgemm_child_count;
    constexpr auto warps_per_block = spgemm_block_size / config::warp_size;
    __shared__ IndexType sh_cols[heap_size * warps_per_block];
    __shared__ IndexType sh_idxs[heap_size * warps_per_block];
    __shared__ IndexType sh_ends[heap_size * warps_per_block];
    __shared__ UninitializedArray<ValueType, heap_size * warps_per_block>
        sh_vals;
    if (a_size <= 2) {
        spgemm_multiway_merge_short<2, use_val>(a_size, a_cols, a_vals,
                                                b_row_ptrs, b_cols, b_vals,
                                                step_cb, col_cb);
    } else if (a_size <= 4) {
        spgemm_multiway_merge_short<4, use_val>(a_size, a_cols, a_vals,
                                                b_row_ptrs, b_cols, b_vals,
                                                step_cb, col_cb);
    } else if (a_size <= 8) {
        spgemm_multiway_merge_short<8, use_val>(a_size, a_cols, a_vals,
                                                b_row_ptrs, b_cols, b_vals,
                                                step_cb, col_cb);
    } else if (a_size <= 16) {
        spgemm_multiway_merge_short<16, use_val>(a_size, a_cols, a_vals,
                                                 b_row_ptrs, b_cols, b_vals,
                                                 step_cb, col_cb);
    } else if (a_size <= 32) {
        spgemm_multiway_merge_short<32, use_val>(a_size, a_cols, a_vals,
                                                 b_row_ptrs, b_cols, b_vals,
                                                 step_cb, col_cb);
    } else if (a_size <= config::warp_size) {
        spgemm_multiway_merge_short<config::warp_size, use_val>(
            a_size, a_cols, a_vals, b_row_ptrs, b_cols, b_vals, step_cb,
            col_cb);
    } else if (a_size <= config::warp_size * (spgemm_child_count + 1)) {
        spgemm_multiway_merge_medium<use_val>(
            a_size, a_cols, a_vals, b_row_ptrs, b_cols, b_vals, sh_cols,
            sh_idxs, sh_ends, &sh_vals[0], step_cb, col_cb);
    } else {
        spgemm_multiway_merge_large<use_val>(
            a_size, a_cols, a_vals, b_row_ptrs, b_cols, b_vals, sh_cols,
            sh_idxs, sh_ends, &sh_vals[0], tmp, tmp + a_size, tmp + 2 * a_size,
            tmpval, step_cb, col_cb);
    }
}


template <typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_count(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ a_cols,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, IndexType *__restrict__ tmp,
    IndexType *__restrict__ c_nnz)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>();
    const auto subwarp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    if (row >= num_rows) {
        return;
    }

    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    IndexType nnz{};
    auto step_cb = [](float, IndexType) {};
    auto col_cb = [&](IndexType) { ++nnz; };
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        auto a_col = a_cols[a_begin];
        nnz = b_row_ptrs[a_col + 1] - b_row_ptrs[a_col];
    } else {
        auto null_val = static_cast<float *>(nullptr);
        spgemm_multiway_merge_dispatch<false>(
            a_size, a_cols + a_begin, null_val, b_row_ptrs, b_cols, null_val,
            tmp + 3 * a_begin, null_val, step_cb, col_cb);
    }
    if (subwarp.thread_rank() == 0) {
        c_nnz[row] = nnz;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spgemm_block_size) void spgemm_kernel(
    size_type num_rows, const IndexType *__restrict__ a_row_ptrs,
    const IndexType *__restrict__ a_cols, const ValueType *__restrict__ a_vals,
    const IndexType *__restrict__ b_row_ptrs,
    const IndexType *__restrict__ b_cols, const ValueType *__restrict__ b_vals,
    const IndexType *__restrict__ c_row_ptrs, IndexType *__restrict__ tmp,
    ValueType *__restrict__ tmpval, IndexType *__restrict__ c_cols,
    ValueType *__restrict__ c_vals)
{
    constexpr auto subwarp_size = config::warp_size;
    const auto row = thread::get_subwarp_id_flat<subwarp_size>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (row >= num_rows) {
        return;
    }

    const auto a_begin = a_row_ptrs[row];
    const auto a_end = a_row_ptrs[row + 1];
    const auto a_size = a_end - a_begin;
    IndexType c_nz = c_row_ptrs[row];
    ValueType c_val{};
    auto step_cb = [&](ValueType val, IndexType) { c_val += val; };
    auto col_cb = [&](IndexType c_col) {
        if (subwarp.thread_rank() == 0) {
            c_cols[c_nz] = c_col;
            c_vals[c_nz] = c_val;
        }
        ++c_nz;
        c_val = zero<ValueType>();
    };
    if (a_size == 0) {
        // do nothing
    } else if (a_size == 1) {
        auto a_col = a_cols[a_begin];
        auto a_val = a_vals[a_begin];
        auto b_begin = b_row_ptrs[a_col];
        auto b_size = b_row_ptrs[a_col + 1] - b_begin;
        for (auto i = subwarp.thread_rank(); i < b_size; i += subwarp_size) {
            c_cols[c_nz + i] = b_cols[b_begin + i];
            c_vals[c_nz + i] = a_val * b_vals[b_begin + i];
        }
    } else {
        spgemm_multiway_merge_dispatch<true>(
            a_size, a_cols + a_begin, a_vals + a_begin, b_row_ptrs, b_cols,
            b_vals, tmp + a_begin * 3, tmpval + a_begin, step_cb, col_cb);
    }
}