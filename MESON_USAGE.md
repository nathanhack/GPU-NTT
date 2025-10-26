# Using GPU-NTT with Meson

## Building GPU-NTT

### Basic build:
```bash
meson setup build -Dgpu_arch=86
meson compile -C build
```

### With examples:
```bash
meson setup build -Dgpu_arch=86 -Dbuild_examples=true
meson compile -C build
```

### Installation:
```bash
meson setup build -Dgpu_arch=86 --prefix=/usr/local
meson compile -C build
sudo meson install -C build
```

## Using GPU-NTT in a downstream Meson project

### Method 1: Using installed library (via pkg-config)

After installing GPU-NTT, you can use it in your project's `meson.build`:

```meson
project('my-project', ['cpp', 'cuda'],
  version: '1.0',
  default_options: ['cpp_std=c++17']
)

# Find CUDA
cuda_dep = dependency('cuda', modules: ['cudart'])

# Find GPUNTT via pkg-config
gpuntt_dep = dependency('gpuntt', version: '>=1.0')

# Your executable
executable('my_app',
  'main.cu',
  dependencies: [cuda_dep, gpuntt_dep],
  cuda_args: ['-arch=sm_86', '--expt-relaxed-constexpr'],
)
```

### Method 2: Using as a subproject

1. Create a `subprojects` directory in your project:
```bash
mkdir -p subprojects
```

2. Either:
   - Clone GPU-NTT into `subprojects/GPUNTT/`, or
   - Create a wrap file at `subprojects/gpuntt.wrap`:

```ini
[wrap-git]
url = https://github.com/Alisah-Ozcan/GPU-NTT.git
revision = main

[provide]
gpuntt = gpuntt_dep
```

3. Use it in your `meson.build`:

```meson
project('my-project', ['cpp', 'cuda'],
  version: '1.0',
  default_options: ['cpp_std=c++17']
)

# Find CUDA
cuda_dep = dependency('cuda', modules: ['cudart'])

# Get GPUNTT as a subproject
gpuntt_dep = dependency('gpuntt', fallback: ['GPUNTT', 'ntt_dep'])

# Your executable
executable('my_app',
  'main.cu',
  dependencies: [cuda_dep, gpuntt_dep],
  cuda_args: ['-arch=sm_86', '--expt-relaxed-constexpr'],
)
```

## CUDA Architecture

Specify your GPU architecture:
- For RTX 30xx series: `-Dgpu_arch=86`
- For RTX 20xx series: `-Dgpu_arch=75`
- For RTX 40xx series: `-Dgpu_arch=89`
- For A100: `-Dgpu_arch=80`

## Exported Headers

When using GPU-NTT, include headers as:

```cpp
#include "ntt.cuh"              // Merge NTT
#include "ntt_cpu.cuh"          // Merge NTT CPU
#include "ntt_4step.cuh"        // 4-Step NTT
#include "ntt_4step_cpu.cuh"    // 4-Step NTT CPU
#include "common.cuh"           // Common utilities
#include "nttparameters.cuh"    // NTT parameters
#include "modular_arith.cuh"    // Modular arithmetic
```

## Exported Functions

Based on the example files, the key exportable functions are:

### From `ntt.cuh` (Merge NTT):
- `GPU_NTT_Inplace()` - In-place forward NTT
- `GPU_NTT()` - Forward NTT with separate input/output
- `GPU_INTT_Inplace()` - In-place inverse NTT
- `GPU_INTT()` - Inverse NTT with separate input/output

### From `ntt_cpu.cuh`:
- `NTTCPU<T>::ntt()` - CPU reference implementation
- `NTTCPU<T>::intt()` - CPU inverse NTT

### Types:
- `Data32` / `Data64` - Unsigned 32/64-bit data types
- `Data32s` / `Data64s` - Signed 32/64-bit data types
- `Root<T>` - Root of unity type
- `Modulus<T>` - Modulus type
- `Ninverse<T>` - N inverse type
- `NTTParameters<T>` - NTT parameter configuration

### Configuration:
- `ntt_configuration<T>` - Single modulus configuration
- `ntt_rns_configuration<T>` - RNS (multi-modulus) configuration
- `ReductionPolynomial` enum - X_N_minus or X_N_plus
- `NTT_LAYOUT` enum - PerPolynomial or PerCoefficient
