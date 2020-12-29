#!/usr/bin/env python
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='torch_discounted_cumsum',
    version=0.1,
    description='Fast discounted cumulative sum in PyTorch',
    install_requires=requirements,
    python_requires='>=3.6',
    packages=find_packages(),
    author='Anton Obukhov',
    license='BSD',
    url='https://www.github.com/toshas/torch-discounted-cumsum',
    ext_modules=[
        CppExtension(
            'torch_discounted_cumsum_cpu',
            [
                os.path.join('torch_discounted_cumsum', 'discounted_cumsum_cpu.cpp'),
            ],
        ),
        CUDAExtension(
            'torch_discounted_cumsum_cuda',
            [
                os.path.join('torch_discounted_cumsum', 'discounted_cumsum_cuda.cpp'),
                os.path.join('torch_discounted_cumsum', 'discounted_cumsum_cuda_kernel.cu'),
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    keywords=[
        'pytorch', 'discounted', 'cumsum', 'cumulative', 'sum', 'scan', 'differentiable',
        'reinforcement', 'learning', 'rewards', 'time', 'series'
    ],
)
