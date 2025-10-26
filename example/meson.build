# Examples
examples = {
  'gpu_merge_ntt_examples': 'ntt_merge/test_merge_ntt.cu',
  'gpu_merge_intt_examples': 'ntt_merge/test_merge_intt.cu',
  'cpu_merge_ntt_examples': 'ntt_merge/test_cpu_merge_ntt.cu',
  'gpu_4step_ntt_examples': 'ntt_4step/test_4step_ntt.cu',
  'gpu_4step_intt_examples': 'ntt_4step/test_4step_intt.cu',
  'cpu_4step_ntt_examples': 'ntt_4step/test_cpu_4step_ntt.cu',
}

foreach name, source : examples
  executable(name,
    source,
    dependencies: [ntt_dep, cuda_dep],
    install: false,
  )
endforeach
