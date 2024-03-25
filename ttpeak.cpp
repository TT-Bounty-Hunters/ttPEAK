
#include "common/core_coord.h"
#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include <cstdint>
#include <memory>
#include <chrono>
#include <vector>
#include <cmath>

using namespace tt::tt_metal;
using namespace chrono;

constexpr size_t experiment_runs = 10;

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

double test_dram_read(Device* device, CommandQueue& cq, long program_latency, size_t n_cores)
{
    if(n_cores == 0)
        throw std::runtime_error("Number of cores must be greater than 0");

    Program program = CreateProgram();

    // Don't care about the contents of the buffer. We just want to measure the bandwidth
    auto dram_buffer = MakeBufferBFP16(device, 512, false);
    auto l1_buffer = MakeBufferBFP16(device, 64, true);
    const size_t core_grid_width = device->logical_grid_size().x;

    for(size_t i = 0; i < n_cores; i++)
    {
        CoreCoord core = {i % core_grid_width, i / core_grid_width};
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
    }

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
    double bandwidth_gbs = (double)dram_buffer->size() / real_duration * n_cores;

    return bandwidth_gbs;
}

double test_elemwise_op(Device* device, CommandQueue& cq, long program_latency)
{
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    KernelHandle movement = CreateKernel(
        program,
        "ttpeak_kernels/movement/generate_dummy.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    KernelHandle compute = CreateKernel(
        program,
        "ttpeak_kernels/compute/unary.cpp",
        core,
        ComputeConfig{
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {}
        }
    );

    constexpr uint32_t n_tiles = 2048;
    constexpr uint32_t single_tile_size = 32 * 32 * 2;
    constexpr uint32_t cb_tiles = 4;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(cb_tiles * single_tile_size, {{tt::CB::c_in0, tt::DataFormat::Float16_b}}).set_page_size(tt::CB::c_in0, single_tile_size);
    CBHandle cb_src0 = CreateCircularBuffer(program, core, cb_src0_config);

    SetRuntimeArgs(program, movement, core, {n_tiles});
    SetRuntimeArgs(program, compute, core, {n_tiles});

    EnqueueProgram(cq, program, false);
    Finish(cq);

    auto t1 = high_resolution_clock::now();
    EnqueueProgram(cq, program, false);
    Finish(cq);
    auto t2 = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(t2 - t1).count();

    size_t flop = n_tiles * 32 * 32;
    double glops = (double)flop / (duration - program_latency);
    
    return glops;

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
    std::cout << "Device info:\n"
        << "  Architecture: " << arch2name(device->arch()) << "\n"
        << "  Device ID: " << device->id() << "\n"
        << "  # of hardware command queues: " << (int)device->num_hw_cqs() << "\n"
        << "  L1 memory per core: " << device->l1_size_per_core() / 1024 << " KiB\n"
        << "  Logical core grid size: " << core_grid.x << "x" << core_grid.y << "\n"
        << "  DRAM banks: " << device->num_banks(BufferType::DRAM) << "\n"
        << "  DRAM bank size: " << device->bank_size(BufferType::DRAM) / 1024 / 1024 << " MiB\n"
        << "  DRAM channels: " << device->num_dram_channels() << "\n"
        << "  DRAM size per channel: " << device->dram_size_per_channel() / 1024 / 1024 << " MiB\n"
        << "  Machine epsilon: " << device->sfpu_eps() << "\n"
        << std::endl;
}


int main(int argc, char **argv)
{
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    print_device_info(device);

    CommandQueue& cq = device->command_queue();

    std::cout << "DRAM Bandwidth:" << std::endl;
    size_t program_run_ns = test_program_run_latency(device, cq);
    double dram_gbs = test_dram_read(device, cq, program_run_ns, 1);
    std::cout << "  DRAM read bandwidth (1 core): " << dram_gbs << " GB/s" << std::endl;
    double dram_gbs_8 = test_dram_read(device, cq, program_run_ns, 8);
    std::cout << "  DRAM read bandwidth (8 cores): " << dram_gbs_8 << " GB/s" << std::endl;

    std::cout << "Compute: " << std::endl;
    double elemwise_glfops = test_elemwise_op(device, cq, program_run_ns);
    std::cout << "  Elementwise operation: " << elemwise_glfops << " GLFLOPS" << std::endl;

    std::cout << "\n";
    auto [download_gbs, upload_gbs] = test_data_transfer(device, cq);
    std::cout << "Transfer bandwidth:\n"
        << "  Download: " << download_gbs << " GB/s\n"
        << "  Upload: " << upload_gbs << " GB/s\n";
    
    std::cout << "\n";
    std::cout << "Empty program latency: " << program_run_ns << " ns" << std::endl;
    CloseDevice(device);

    return 0;
}
