# Collect source files
lib_sources = files(
  'lib/common/common.cu',
  'lib/common/nttparameters.cu',
  'lib/ntt_4step/ntt_4step.cu',
  'lib/ntt_4step/ntt_4step_cpu.cu',
  'lib/ntt_merge/ntt.cu',
  'lib/ntt_merge/ntt_cpu.cu',
)

# Collect header files for installation
lib_headers_common = files(
  'include/common/common.cuh',
  'include/common/modular_arith.cuh',
  'include/common/nttparameters.cuh',
)

lib_headers_ntt4step = files(
  'include/ntt_4step/ntt_4step.cuh',
  'include/ntt_4step/ntt_4step_cpu.cuh',
)

lib_headers_nttmerge = files(
  'include/ntt_merge/ntt.cuh',
  'include/ntt_merge/ntt_cpu.cuh',
)

# Include directories
inc_common = include_directories('include/common')
inc_ntt4step = include_directories('include/ntt_4step')
inc_nttmerge = include_directories('include/ntt_merge')

# Build static library
ntt_lib = static_library('ntt',
  lib_sources,
  include_directories: [inc_common, inc_ntt4step, inc_nttmerge],
  dependencies: [cuda_dep],
  install: true,
)

# Install headers maintaining directory structure
install_headers(lib_headers_common,
  subdir: 'GPUNTT-' + meson.project_version() + '/common')
install_headers(lib_headers_ntt4step,
  subdir: 'GPUNTT-' + meson.project_version() + '/ntt_4step')
install_headers(lib_headers_nttmerge,
  subdir: 'GPUNTT-' + meson.project_version() + '/ntt_merge')

# Declare dependency for downstream projects
ntt_dep = declare_dependency(
  link_with: ntt_lib,
  include_directories: [inc_common, inc_ntt4step, inc_nttmerge],
  dependencies: [cuda_dep],
)

# Make it importable
meson.override_dependency('gpuntt', ntt_dep)
