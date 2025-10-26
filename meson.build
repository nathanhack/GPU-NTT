project('GPUNTT', ['c', 'cpp', 'cuda'],
  version: '1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=2',
  ]
)

# CUDA architecture - can be overridden with -Dgpu_arch=XX
gpu_arch = get_option('gpu_arch')
if gpu_arch == ''
  gpu_arch = '75'
endif
message('CUDA Architecture: ' + gpu_arch)

# Add CUDA architecture flags
cuda_args = ['-arch=sm_' + gpu_arch, '--expt-relaxed-constexpr', '-std=c++17']
add_project_arguments(cuda_args, language: 'cuda')

# Dependencies
cuda_dep = dependency('cuda', version: '>=11.0', modules: ['cudart'])

# Build the library
subdir('src')

# Export package info for downstream projects
pkg = import('pkgconfig')
pkg.generate(
  ntt_lib,
  name: 'GPUNTT',
  description: 'GPU-accelerated Number Theoretic Transform library',
  version: meson.project_version(),
  filebase: 'gpuntt',
  subdirs: ['GPUNTT-' + meson.project_version()],
)

# Examples and benchmarks (optional)
if get_option('build_examples')
  subdir('example')
endif

if get_option('build_benchmarks')
  subdir('benchmark')
endif
