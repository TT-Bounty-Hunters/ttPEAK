#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t in0 = tt::CB::c_in0;
    constexpr uint32_t in1 = tt::CB::c_in1;
    constexpr uint32_t out0 = tt::CB::c_out0;
    binary_op_init_common(in0, in1, out0);
    add_tiles_init();

    for(uint32_t i = 0; i < n_tiles; i+=8) {
        cb_wait_front(in0, 8);
        cb_wait_front(in1, 8);
        // Don't care about the output. discard it.
        tile_regs_acquire(); // acquire 8 tile registers
        for(uint32_t j = 0; j < 8; j++) {
            add_tiles(in0, in1, i, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
        cb_pop_front(in0, 8);
        cb_pop_front(in1, 8);
    }
}
}