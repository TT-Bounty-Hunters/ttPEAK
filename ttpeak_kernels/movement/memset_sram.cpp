#include <cstdint>
#include <cstring>

void kernel_main()
{
    std::uint32_t n_runs        = get_arg_val<uint32_t>(0);

    constexpr uint32_t in0 = tt::CB::c_in0;
    uint32_t tile_size = get_tile_size(in0);

    cb_reserve_back(in0, 8);
    for(uint32_t i = 0; i < n_runs; i++) {
        uint32_t in0_addr = get_write_ptr(in0);
        void* in0_ptr = (void*)in0_addr;
        memset(in0_ptr, 0, tile_size * 8);
    }
    cb_push_back(in0, 8);
}
