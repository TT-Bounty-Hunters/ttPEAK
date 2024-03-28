
#include "common/core_coord.h"
#include "common/logger.hpp"
#include "common/tt_backend_api_types.hpp"
#include "impl/buffers/circular_buffer_types.hpp"
#include "impl/kernels/kernel_types.hpp"
#include "impl/program/program.hpp"
#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include <cstddef>
#include <cstdint>
#include <ios>
#include <memory>
#include <chrono>
#include <sstream>
#include <string_view>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace tt::tt_metal;
using namespace chrono;

using CoreSpec = const std::variant<CoreCoord, CoreRange, CoreRangeSet>;

constexpr size_t experiment_runs = 6;

// Code from stackoverflow
double geometric_mean(const std::vector<double>& data)
{
    long long ex = 0;
    auto do_bucket = [&data,&ex](int first,int last) -> double
    {
        double ans = 1.0;
        for ( ;first != last;++first)
        {
            int i;
            ans *= std::frexp(data[first],&i);
            ex+=i;
        }
        return ans;
    };

    const int bucket_size = -std::log2( std::numeric_limits<double>::min() );
    std::size_t buckets = data.size() / bucket_size;

    double invN = 1.0 / data.size();
    double m = 1.0;

    for (std::size_t i = 0;i < buckets;++i)
        m *= std::pow( do_bucket(i * bucket_size,(i+1) * bucket_size),invN );

    m*= std::pow( do_bucket( buckets * bucket_size, data.size() ),invN );

    return std::pow( std::numeric_limits<double>::radix,ex * invN ) * m;
}

std::shared_ptr<Buffer> MakeBuffer(Device *device, uint32_t size, uint32_t page_size, bool sram)
{
    InterleavedBufferConfig config{
        .device= device,
        .size = size,
        .page_size = page_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)
    };
    return CreateBuffer(config);
}

std::shared_ptr<Buffer> MakeBufferBFP16(Device *device, uint32_t n_tiles, bool sram)
{
    constexpr uint32_t tile_size = 2 * (32 * 32);
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeBuffer(device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

CBHandle MakeCircularBuffer(Program& program, const CoreSpec& core, tt::CB cb, uint32_t size, uint32_t page_size, tt::DataFormat format)
{
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
        size,
        {{
            cb,
            format
    }})
    .set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

CBHandle MakeCircularBufferBFP16(Program& program, const CoreSpec& core, tt::CB cb, uint32_t n_tiles)
{
    constexpr uint32_t tile_size = 2 * (32 * 32);
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float16_b);
}

size_t test_program_run_latency(Device* device, CommandQueue& cq)
{
    Program program = CreateProgram();

    constexpr CoreCoord core = {0, 0};

    KernelHandle kernel = CreateKernel(
        program,
        "ttpeak_kernels/movement/noop.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    const std::vector<uint32_t> runtime_args = { 0, 1, 2, 3, 4};
    SetRuntimeArgs(program, kernel, core, runtime_args);

    EnqueueProgram(cq, program, false);
    Finish(cq);

    std::vector<double> runtimes;
    for(size_t i = 0; i < experiment_runs; i++)
    {
        auto t1 = high_resolution_clock::now();
        EnqueueProgram(cq, program, false);
        Finish(cq);
        auto t2 = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(t2 - t1).count();
        runtimes.push_back(duration);
    }
    return geometric_mean(runtimes);
}

double test_dram_read(Device* device, CommandQueue& cq, long program_latency, CoreRange core = CoreCoord{0, 0})
{
    Program program = CreateProgram();

    // Don't care about the contents of the buffer. We just want to measure the bandwidth
    // Each tile is 2 * (32 * 32) = 2 KiB. 1K tiles = 2 MiB
    auto dram_buffer = MakeBufferBFP16(device, 8 * 1024, false);
    auto l1_buffer = MakeBufferBFP16(device, 32, true);
    const size_t data_size = dram_buffer->size();

    KernelHandle kernel = CreateKernel(
        program,
        "ttpeak_kernels/movement/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    const std::vector<uint32_t> runtime_args = {
        l1_buffer->address(),
        l1_buffer->size(),
        dram_buffer->address(),
        (uint32_t) dram_buffer->noc_coordinates().x,
        (uint32_t) dram_buffer->noc_coordinates().y,
        dram_buffer->size(),
    };

    SetRuntimeArgs(program, kernel, core, runtime_args);

    EnqueueProgram(cq, program, false);
    Finish(cq);

    std::vector<double> runtimes;
    for(size_t i = 0; i < experiment_runs; i++)
    {
        auto t1 = high_resolution_clock::now();
        EnqueueProgram(cq, program, false);
        Finish(cq);
        auto t2 = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(t2 - t1).count();
        runtimes.push_back(duration);
    }

    auto real_duration = geometric_mean(runtimes) - program_latency;
    double bandwidth_gbs = (double)data_size / real_duration * core.size();

    return bandwidth_gbs;
}

double test_noc_bandwidth(Device* device, CommandQueue& cq, long program_latency, bool read)
{
    Program program = CreateProgram();
    constexpr CoreCoord src_core = {0, 0};
    constexpr CoreCoord dst_core = {1, 0};
    KernelHandle writer = CreateKernel(
        program,
        read ? "ttpeak_kernels/movement/noc_reader.cpp" : "ttpeak_kernels/movement/noc_writer.cpp",
        src_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    const uint32_t n_tiles_per_transfer = 32;
    const uint32_t n_transfer = 128;
    const uint32_t tile_size_bytes = 2 * 32 * 32;
    auto cb_src0 = MakeCircularBufferBFP16(program, src_core, tt::CB::c_in0, n_tiles_per_transfer);
    auto cb_src1 = MakeCircularBufferBFP16(program, dst_core, tt::CB::c_in1, n_tiles_per_transfer);

    CoreCoord dst_core_coord = device->worker_core_from_logical_core(dst_core);

    SetRuntimeArgs(program, writer, src_core, {
        L1_UNRESERVED_BASE + n_tiles_per_transfer * tile_size_bytes, // You gotta be kidding me. Manual address calculation?
        (uint32_t)dst_core_coord.x,
        (uint32_t)dst_core_coord.y,
        n_tiles_per_transfer,
        n_transfer
    });

    EnqueueProgram(cq, program, false);
    Finish(cq);
    
    std::vector<double> runtimes;
    for(size_t i = 0; i < experiment_runs; i++)
    {
        auto t1 = high_resolution_clock::now();
        EnqueueProgram(cq, program, false);
        Finish(cq);
        auto t2 = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(t2 - t1).count();
        runtimes.push_back(duration);
    }

    auto real_duration = geometric_mean(runtimes) - program_latency;
    size_t transtered_size = n_transfer * n_tiles_per_transfer * tile_size_bytes;
    double bandwidth_gbs = (double)transtered_size / real_duration;

    return bandwidth_gbs;
}

double test_matmul(Device* device, CommandQueue& cq, long program_latency, CoreRange core = CoreCoord{0, 0})
{
    Program program = CreateProgram();
    KernelHandle movement = CreateKernel(
        program,
        "ttpeak_kernels/movement/generate_dummy_interleaved.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    KernelHandle compute = CreateKernel(
        program,
        "ttpeak_kernels/compute/matmul.cpp",
        core,
        ComputeConfig{
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {}
        }
    );

    constexpr uint32_t n_tiles = 2048; // Must be a multiple of 8
    constexpr uint32_t cb_tiles = 16; // Must be a multiple of 8
    CBHandle cb0 = MakeCircularBufferBFP16(program, core, tt::CB::c_in0, cb_tiles);
    CBHandle cb1 = MakeCircularBufferBFP16(program, core, tt::CB::c_in1, cb_tiles);
    CBHandle cb_out = MakeCircularBufferBFP16(program, core, tt::CB::c_out0, cb_tiles);

    SetRuntimeArgs(program, movement, core, {n_tiles});
    SetRuntimeArgs(program, compute, core, {n_tiles});

    EnqueueProgram(cq, program, false);
    Finish(cq);

    std::vector<double> runtimes;
    for(size_t i = 0; i < experiment_runs; i++)
    {
        auto t1 = high_resolution_clock::now();
        EnqueueProgram(cq, program, false);
        Finish(cq);
        auto t2 = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(t2 - t1).count();
        runtimes.push_back(duration);
    }
    auto real_duration = geometric_mean(runtimes) - program_latency;

    const size_t matmul_flop = n_tiles * 32 * 32 * (2 * 32 - 1) * core.size();
    double gflops = (double)matmul_flop / (real_duration - program_latency);
    
    return gflops;
}

double test_element_wise(Device* device, CommandQueue& cq, long program_latency, CoreRange core = CoreCoord{0, 0})
{
    Program program = CreateProgram();

    auto core_grid = device->compute_with_storage_grid_size();
    constexpr uint32_t n_tiles = 2048; // needs to be a multiple of 8
    constexpr uint32_t cb_tiles = 16; // needs to be a multiple of 8
    KernelHandle movement = CreateKernel(
        program,
        "ttpeak_kernels/movement/generate_dummy_interleaved.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    KernelHandle compute = CreateKernel(
        program,
        "ttpeak_kernels/compute/element_wise.cpp",
        core,
        ComputeConfig{
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {}
        }
    );

    CBHandle cb0 = MakeCircularBufferBFP16(program, core, tt::CB::c_in0, cb_tiles);
    CBHandle cb1 = MakeCircularBufferBFP16(program, core, tt::CB::c_in1, cb_tiles);
    CBHandle cb_out = MakeCircularBufferBFP16(program, core, tt::CB::c_out0, cb_tiles);

    SetRuntimeArgs(program, movement, core, {n_tiles});
    SetRuntimeArgs(program, compute, core, {n_tiles});

    EnqueueProgram(cq, program, false);
    Finish(cq);

    std::vector<double> runtimes;
    for(size_t i = 0; i < experiment_runs; i++)
    {
        auto t1 = high_resolution_clock::now();
        EnqueueProgram(cq, program, false);
        Finish(cq);
        auto t2 = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(t2 - t1).count();
        runtimes.push_back(duration);
    }
    auto real_duration = geometric_mean(runtimes) - program_latency;

    const size_t matmul_flop = n_tiles * core.size() * 32 * 32;
    double gflops = (double)matmul_flop / (real_duration - program_latency);
    
    return gflops;
}

double test_sram_scalar_bandwidth(Device* device, CommandQueue& cq, long program_latency, CoreRange core = CoreCoord{0, 0})
{
    Program program = CreateProgram();

    CBHandle cb0 = MakeCircularBufferBFP16(program, core, tt::CB::c_in0, 8);
    const size_t data_size = 8 * 2 * 32 * 32;

    KernelHandle kernel = CreateKernel(
        program,
        "ttpeak_kernels/movement/memset_sram.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    constexpr uint32_t n_runs = 128;

    const std::vector<uint32_t> runtime_args = {
        n_runs
    };

    SetRuntimeArgs(program, kernel, core, runtime_args);

    EnqueueProgram(cq, program, false);
    Finish(cq);

    std::vector<double> runtimes;
    for(size_t i = 0; i < experiment_runs; i++)
    {
        auto t1 = high_resolution_clock::now();
        EnqueueProgram(cq, program, false);
        Finish(cq);
        auto t2 = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(t2 - t1).count();
        runtimes.push_back(duration);
    }

    auto real_duration = geometric_mean(runtimes) - program_latency;
    double bandwidth_gbs = (double)data_size * n_runs / real_duration * core.size();

    return bandwidth_gbs;
}

std::pair<double, double> test_data_transfer(Device* device, CommandQueue& cq)
{
    auto buf = MakeBufferBFP16(device, 512, false);
    std::vector<uint32_t> dummy_data(buf->size() / 4);

    Finish(cq);

    auto t1 = high_resolution_clock::now();
    EnqueueWriteBuffer(cq, buf, dummy_data, false);
    Finish(cq);
    auto t2 = high_resolution_clock::now();
    auto write_duration = duration_cast<nanoseconds>(t2 - t1).count();

    t1 = high_resolution_clock::now();
    EnqueueReadBuffer(cq, buf, dummy_data, false);
    Finish(cq);
    t2 = high_resolution_clock::now();
    auto read_duration = duration_cast<nanoseconds>(t2 - t1).count();

    double read_bandwidth_gbs = (double)buf->size() / read_duration;
    double write_bandwidth_gbs = (double)buf->size() / write_duration;

    return {read_bandwidth_gbs, write_bandwidth_gbs};
}

std::string arch2name(tt::ARCH arch)
{
    switch (arch)
    {
        case tt::ARCH::JAWBRIDGE:
            return "JAWBRIDGE";
        case tt::ARCH::GRAYSKULL:
            return "GRAYSKULL";
        case tt::ARCH::WORMHOLE:
            return "WORMHOLE";
        case tt::ARCH::WORMHOLE_B0:
            return "WORMHOLE_B0";
        case tt::ARCH::BLACKHOLE:
            return "BLACKHOLE";
        case tt::ARCH::Invalid:
            return "Invalid";
    }
    return "Unknown";
}

void print_device_info(Device* device)
{
    CoreCoord core_grid = device->logical_grid_size();
    CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    std::cout << "Device info:\n"
        << "  Architecture                    : " << arch2name(device->arch()) << "\n"
        << "  Device ID                       : " << device->id() << "\n"
        << "  # of hardware command queues    : " << (int)device->num_hw_cqs() << "\n"
        << "  L1 memory per core              : " << device->l1_size_per_core() / 1024 << " KiB\n"
        << "  Logical core grid size          : " << core_grid.x << "x" << core_grid.y << "\n"
        << "  Compute with storage grid size  : " << compute_with_storage_grid_size.x << "x" << compute_with_storage_grid_size.y << "\n"
        << "  DRAM banks                      : " << device->num_banks(BufferType::DRAM) << "\n"
        << "  DRAM bank size                  : " << device->bank_size(BufferType::DRAM) / 1024 / 1024 << " MiB\n"
        << "  DRAM channels                   : " << device->num_dram_channels() << "\n"
        << "  DRAM size per channel           : " << device->dram_size_per_channel() / 1024 / 1024 << " MiB\n"
        << "  Machine epsilon                 : " << device->sfpu_eps() << "\n"
        << std::endl;
}

void help(std::string_view arg0)
{
    std::cout << "Usage: " << arg0 << " [-d <device_id>]\n"
        << "Synthetic benchmarking tool to measure peak capabilities of Tenstorrent devices.\n"
        << "\n"
        << "  -d <device_id>  Device ID to run the test on\n"
        << std::endl;
}

std::string next_arg(int& i, int argc, char** argv)
{
    if(i + 1 >= argc)
        throw std::runtime_error("Expected argument after " + std::string(argv[i]));
    return argv[++i];
}

std::string pretty_gflops(double gflops)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    if(gflops < 1)
        ss << gflops * 1000 << " MFLOPS";
    else if (gflops > 1000)
        ss << gflops / 1000 << " TFLOPS";
    else
        ss << gflops << " GFLOPS";
    return ss.str();
}

int main(int argc, char **argv)
{
    int device_id = 0;
    for(int i = 1; i < argc; i++)
    {
        std::string_view arg = argv[i];
        if(arg == "-d")
            device_id = std::stoi(next_arg(i, argc, argv));
        else if(arg == "-h" || arg == "--help") {
            help(argv[0]);
            return 0;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            help(argv[0]);
            return 1;
        
        }

    }

    Device *device = CreateDevice(device_id);
    print_device_info(device);
    device->enable_program_cache();

    CommandQueue& cq = device->command_queue();
    auto core_grid = device->compute_with_storage_grid_size();
    auto all_cores = CoreRange(CoreCoord{0, 0}, CoreCoord{core_grid.x-1, core_grid.y-1});

    std::cout << "Bandwidth (GB/s):" << std::endl;
    std::streamsize ss = std::cout.precision();
    std::cout << std::fixed << std::setprecision(2);
    size_t program_run_ns = test_program_run_latency(device, cq);
    double dram_gbs = test_dram_read(device, cq, program_run_ns);
    std::cout << "  DRAM read bandwidth (1 core)     : " << dram_gbs << std::endl;
    double dram_gbs_all = test_dram_read(device, cq, program_run_ns, all_cores);
    std::cout << "  DRAM read bandwidth (all cores)  : " << dram_gbs_all  << std::endl;
    double adjacent_write = test_noc_bandwidth(device, cq, program_run_ns, false);
    std::cout << "  Adjacent core NoC write          : " << adjacent_write << std::endl;
    double adjacent_read = test_noc_bandwidth(device, cq, program_run_ns, true);
    std::cout << "  Adjacent core NoC read           : " << adjacent_read  << std::endl;
    double sram_scalar_gbs = test_sram_scalar_bandwidth(device, cq, program_run_ns);
    std::cout << "  SRAM scalar r/w (per core)       : " << sram_scalar_gbs << " (this is expected to be slow)" << std::endl;

    std::cout << "\n";
    std::cout << "Compute (BFP16, GFLOPS): " << std::endl;
    double element_wise_gflops = test_element_wise(device, cq, program_run_ns);
    std::cout << "  Element wise math (1 core)       : " << element_wise_gflops << std::endl;
    double element_wise_gflops_all = test_element_wise(device, cq, program_run_ns, all_cores);
    std::cout << "  Element wise math (all cores)    : " << element_wise_gflops_all << std::endl;
    double matmul_gflops = test_matmul(device, cq, program_run_ns);
    std::cout << "  Matrix multiplcation (1 core)    : " << matmul_gflops << std::endl;
    double matmul_gflops_all = test_matmul(device, cq, program_run_ns, all_cores);
    std::cout << "  Matrix multiplcation (all cores) : " << matmul_gflops_all << std::endl;

    std::cout << "\n";
    auto [download_gbs, upload_gbs] = test_data_transfer(device, cq);
    std::cout << "Transfer bandwidth (GB/s):\n"
        << "  Download          : " << download_gbs << "\n"
        << "  Upload            : " << upload_gbs << "\n";
    
    std::cout << "\n";
    std::cout << "Empty program launch latency: " << program_run_ns << " ns" << std::endl;
    std::cout << std::setprecision(ss);
    CloseDevice(device);

    return 0;
}
