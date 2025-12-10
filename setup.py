from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools.command.egg_info import egg_info

BIND_PATH = "./backend/cuda/bindings/"
OP_PATH = "./backend/cuda/ops/"


setup(
    name="ops",
    ext_modules=[
        CUDAExtension(
            name="vector_add",
            sources=[
                BIND_PATH + "/vector_add/vector_add.cpp",
                OP_PATH + "/vector_add/vector_add_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        ),
        CUDAExtension(
            name="reduce",
            sources=[
                BIND_PATH + "/reduce/reduce.cpp",
                OP_PATH + "/reduce/reduce_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        ),
        CUDAExtension(
            name="gemm",
            sources=[
                BIND_PATH + "/gemm/gemm.cpp",
                OP_PATH + "/gemm/gemm_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        ),
        CUDAExtension(
            name="attention",
            sources=[
                BIND_PATH + "/attention/attention.cpp",
                OP_PATH + "/attention/attention_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        )
    ],
    options={
        "egg_info": {
            "egg_base": "./build",
        }
    },
    cmdclass={
        "build_ext": BuildExtension,
    },
)
