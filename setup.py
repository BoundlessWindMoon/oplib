from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools.command.egg_info import egg_info

BIND_PATH = "./core/bindings/"
OP_PATH = "./core/ops/"


class DisableEggInfo(egg_info):
    def run(self):
        pass


setup(
    name="ops",
    ext_modules=[
        CUDAExtension(
            name="vector_add",
            sources=[
                BIND_PATH + "/vector_add/vector_add.cpp",
                OP_PATH + "/vector_add/vector_add_kernel.cu",
            ],
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
