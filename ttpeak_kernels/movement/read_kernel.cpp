#include <cstdint>

void kernel_main()
{
    std::uint32_t l1_addr        = get_arg_val<uint32_t>(0);
    std::uint32_t l1_size        = get_arg_val<uint32_t>(1);

    std::uint32_t dram_addr      = get_arg_val<uint32_t>(2);
    std::uint32_t dram_noc_x     = get_arg_val<uint32_t>(3);
    std::uint32_t dram_noc_y     = get_arg_val<uint32_t>(4);
    std::uint32_t dram_size      = get_arg_val<uint32_t>(5);

    if(l1_size == 0)
        return;

    const uint32_t tile_size_bytes = 32 * 32 * 2;

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = dram_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b,
    };

    uint32_t dram_tiles = dram_size / tile_size_bytes;
    uint32_t l1_tiles = l1_size / tile_size_bytes;

    for(uint32_t i = 0; i < dram_tiles; i += l1_tiles) {

        for(uint32_t j = 0; j < l1_tiles; j += 1) {
            uint32_t read_to = l1_addr + j * tile_size_bytes;
            noc_async_read_tile(i+j, s, read_to);
        }
        noc_async_read_barrier();
    } 
}
