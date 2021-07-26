#!/usr/bin/env python
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


long_description = """
This package implements an efficient parallel algorithm for the computation of
mutual information between sequences with differentiable bindings to PyTorch.


Find more details and the most up-to-date information on the project webpage:
[TODO]
"""


def configure_extensions():
    out = [
        CppExtension(
            'torch_mutual_information_cpu',
            [
                os.path.join('torch_mutual_information', 'mutual_information_cpu.cpp'),
            ],
        )
    ]
    try:
        out.append(
            CUDAExtension(
                'torch_mutual_information_cuda',
                [
                    os.path.join('torch_mutual_information', 'mutual_information_cuda.cpp'),
                    os.path.join('torch_mutual_information', 'mutual_information_cuda_kernel.cu'),
                ],
            )
        )
    except Exception as e:
        print(f'Failed to build CUDA extension, this part of the package will not work. Reason: {str(e)}')
    return out


setup(
    name='torch_mutual_information',
    version='1.0.2',
    description='Mutual information between sequences of vectors',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    python_requires='>=3.6',
    packages=find_packages(),
    author='Dan Povey',
    license='BSD',
    ext_modules=configure_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    },
    keywords=[
        'pytorch', 'sequence', 'mutual', 'information'
    ],
)
