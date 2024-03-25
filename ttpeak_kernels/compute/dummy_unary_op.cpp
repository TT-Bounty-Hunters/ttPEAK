#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/relu.h"

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);


    for(uint32_t i = 0; i < n_tiles; i++) {
        acquire_dst(tt::DstMode::Half);
        cb_wait_front(tt::CB::c_in0, 1);
        // copy_tile(tt::CB::c_in0, 0, 0); // hangs the hardware??

        relu_tile_init();
        relu_tile(0);

        cb_pop_front(tt::CB::c_in0, 1);
        release_dst(tt::DstMode::Half);
    }
}
}