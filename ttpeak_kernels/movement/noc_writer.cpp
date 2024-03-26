#include <stdint.h>

#include "dataflow_api.h"

void kernel_main()
{
    uint32_t dst_noc_cb1 = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_x = get_arg_val<uint32_t>(1);
    uint32_t dst_noc_y = get_arg_val<uint32_t>(2);
    uint32_t n_tiles_per_transfer = get_arg_val<uint32_t>(3);
    uint32_t n_transfer = get_arg_val<uint32_t>(4);


    constexpr uint32_t cb0_id = 0;
    uint32_t transfer_size = get_tile_size(cb0_id) * n_tiles_per_transfer;

    uint32_t cb0_addr;
    cb_reserve_back(cb0_id, 1);
    cb0_addr = get_write_ptr(cb0_id);

    for (uint32_t i = 0; i < n_transfer; i++) {
        uint64_t target_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_noc_cb1);

        noc_async_write(cb0_addr, target_noc_addr, transfer_size);
        noc_async_write_barrier();
    }
}
