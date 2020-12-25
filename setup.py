from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torch_discounted_cumsum',
    ext_modules=[
        CUDAExtension('lltm_cuda', [
            'discounted_cumsum.cpp',
            'discounted_cumsum_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
