#include <cstdint>

void kernel_main()
{
    std::uint32_t n_tiles        = get_arg_val<uint32_t>(0);

    constexpr uint32_t in0 = tt::CB::c_in0;
    constexpr uint32_t in1 = tt::CB::c_in1;

    for(uint32_t i = 0; i < n_tiles/8; i++) {
        // Creats 8 tiles out of thin air. Tile content will be garbage.
        // 8 because this is faster.
        cb_reserve_back(in0, 8);
        cb_reserve_back(in1, 8);

        cb_push_back(in0, 8);
        cb_push_back(in1, 8);
    }
}
