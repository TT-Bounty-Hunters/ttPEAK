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
2024-03-25 04:26:51.495 | INFO     | SiliconDriver   - Detected 1 PCI device : [0]
                  Metal | INFO     | AI CLK for device 0 is:   1300 MHz
Device info:
  Architecture: GRAYSKULL
  Device ID: 0
  # of hardware command queues: 1
  L1 memory per core: 1024 KiB
  Logical core grid size: 12x8
  DRAM banks: 8
  DRAM bank size: 1023 MiB
  DRAM channels: 8
  DRAM size per channel: 1024 MiB
  Machine epsilon: 0.00195312

DRAM Bandwidth:
  DRAM read bandwidth (1 core): 15.4251 GB/s
  DRAM read bandwidth (8 cores): 12.0695 GB/s
Compute: 
  Elementwise operation: 5.49921 GLFLOPS

Transfer bandwidth:
  Download: 3.3803 GB/s
  Upload: 1.13151 GB/s

Empty program latency: 64311 ns
                  Metal | INFO     | Closing device 0
```