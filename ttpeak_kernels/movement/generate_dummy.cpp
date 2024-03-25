#include <cstdint>

void kernel_main()
{
    std::uint32_t n_tiles        = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in0 = 0;
    for(uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_id_in0, 1);
        // uint32_t write_addr = get_write_ptr(cb_id_in0);
        // No-op. Don't care about the value.
        cb_push_back(cb_id_in0, 1);
    }
}
