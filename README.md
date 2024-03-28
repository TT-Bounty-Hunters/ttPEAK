# ttPEAK

A synthetic benchmarking tool to measure peak capabilities of Tenstorrent devices. It only measues the peak metrics that can be easily measured and easy to program for. It does not represent the actual performance of the device in real world applications.


## How to build

**This section needs to be cleaned up and provide clearer instructions**

Before building this project. Setup the enviroment variables that Metalium needs

```bash
export ARCH_NAME=grayskull                                                                   
export TT_METAL_HOME=/path/to/your/root/of/tt-metal/
export PYTHONPATH=$TT_METAL_HOME
export TT_METAL_ENV=dev
```

Then build the project

```bash
cd ttPEAK
# HACK: Metalium only searches kernels under it's directories
ln -s ttpeak_kernels $TT_METAL_HOME/ttpeak_kernels

mkdir build
cd build
cmake ..
make
```

To run

```
➜ ./ttpeak 
                  Metal | INFO     | Initializing device 0
                 Device | INFO     | Opening user mode device driver
2024-03-28 08:47:29.836 | INFO     | SiliconDriver   - Detected 1 PCI device : {0}
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
  DRAM read bandwidth (1 core)     : 24.2766
  DRAM read bandwidth (all cores)  : 67.2764
  Adjacent core NoC write          : 38.1825
  Adjacent core NoC read           : 38.4963
  SRAM scalar r/w (per core)       : 0.980124 (this is expected to be slow)

Compute (BFP16, GFLOPS): 
  Matrix multiplcation (1 core)    : 943.583
  Matrix multiplcation (all cores) : 79188
  Element wise math (1 core)       : 98.0736
  Element wise math (all cores)    : 8215.74

Transfer bandwidth (GB/s):
  Download          : 2.60739
  Upload            : 1.12218

Empty program launch latency: 64043 ns
                  Metal | INFO     | Closing device 0
                 Device | INFO     | Closing user mode device drivers
```