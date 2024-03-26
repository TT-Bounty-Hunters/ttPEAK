#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t in0 = tt::CB::c_in0;
    constexpr uint32_t in1 = tt::CB::c_in1;
    constexpr uint32_t out0 = tt::CB::c_out0;
    // binary_op_init_common(in0, in1, out0);
    mm_init(in0, in1, out0);

    for(uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(in0, 1);
        cb_wait_front(in1, 1);
        // Don't care about the output. discard it.
        acquire_dst(tt::DstMode::Half);
        matmul_tiles(in0, in1, 0, 0, 0, false);

        release_dst(tt::DstMode::Half);
        cb_pop_front(in0, 1);
        cb_pop_front(in1, 1);
    }
}
}