# ttPEAK

A synthetic benchmarking tool to measure peak capabilities of Tenstorrent devices. It only measues the peak metrics that can be easily measured and easy to program for. It does not represent the actual performance of the device in real world applications.


## How to build

**This section needs to be cleaned up and provide clearer instructions**

Before building this project. Setup the enviroment variables that Metalium needs

```bash
export ARCH_NAME=grayskull                                                                   
export TT_METAL_HOME=/path/to/your/root/of/tt-metal/
export TT_METAL_ENV=dev
```

Then build the project

```bash
cd ttPEAK
mkdir build
cd build
# Current tt-metal uses clang and libc++
cmake . -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-stdlib=libc++" -DCMAKE_EXE_LINKER_FLAGS="-lc++abi -lc++"
make
# Link the kernel directory. see bug https://github.com/tenstorrent/tt-metal/issues/9470
ln -s `pwd`/../ttpeak_kernels $TT_METAL_HOME/ttpeak_kernels
```

To run the program (the output is from my system with a Grayskull e75 on x4 PCIe Gen4)

```
➜ cd .. # Move to the root of the project. NOT the build directory
➜ build/ttpeak 
                  Metal | INFO     | Initializing device 0
                 Device | INFO     | Opening user mode device driver
2024-03-28 08:52:16.488 | INFO     | SiliconDriver   - Detected 1 PCI device : {0}
                  Metal | INFO     | AI CLK for device 0 is:   1300 MHz
Device info:
  Architecture                    : GRAYSKULL
  Device ID                       : 0
  # of hardware command queues    : 1
  L1 memory per core              : 1024 KiB
  Logical core grid size          : 12x8
  Compute with storage grid size  : 11x8
  DRAM banks                      : 8
  DRAM bank size                  : 1023 MiB
  DRAM channels                   : 8
  DRAM size per channel           : 1024 MiB
  Machine epsilon                 : 0.00195312

Bandwidth (GB/s):
  DRAM read bandwidth (1 core)     : 22.99
  DRAM read bandwidth (all cores)  : 67.45
  Adjacent core NoC write          : 38.15
  Adjacent core NoC read           : 38.47
  SRAM scalar r/w (per core)       : 0.98 (this is expected to be slow)

Compute (BFP16, GFLOPS): 
  Element wise math (1 core)       : 113.32
  Element wise math (all cores)    : 7926.89
  Matrix multiplcation (1 core)    : 940.12
  Matrix multiplcation (all cores) : 64805.11

Transfer bandwidth (GB/s):
  Download          : 2.77
  Upload            : 1.13

Empty program launch latency: 63793 ns
                  Metal | INFO     | Closing device 0
                 Device | INFO     | Closing user mode device drivers
```
