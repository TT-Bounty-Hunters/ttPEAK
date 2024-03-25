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

    for(uint32_t i = 0; i < dram_size; i += l1_size) {
        std::uint64_t dram_buffer_src_noc_addr = get_noc_addr(dram_noc_x, dram_noc_y, dram_addr + i);
        noc_async_read(dram_buffer_src_noc_addr, l1_addr, std::min(l1_size, dram_size - i));
        noc_async_read_barrier();
    } 
}
