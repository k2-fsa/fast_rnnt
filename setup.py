from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='torch_discounted_cumsum',
    ext_modules=[
        CppExtension('torch_discounted_cumsum_cpu', [
            'discounted_cumsum_cpu.cpp'
        ]),
        CUDAExtension('torch_discounted_cumsum_cuda', [
            'discounted_cumsum_cuda.cpp',
            'discounted_cumsum_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
