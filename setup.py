import itertools
import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


# environment specific requirements
extras = {
    "games": ["pygame>=2.1.2", "ruamel.yaml>=0.17.21"],
    "reinforcement-learning": ["gym>=0.25", "opencv-python"],
    "simple-multirotor": ["ruamel.yaml>=0.17.21"],
}  # NOQA

extras["reinforcement-learning"].extend(
    [r for r in extras["games"] if r not in extras["reinforcement-learning"]]
)
extras["all"] = list(
    itertools.chain.from_iterable(map(lambda group: extras[group], extras.keys()))
)


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="gncpy",
    version="0.0.0",
    description="A package for Guidance, Navigation, and Control (GNC) algorithms.",
    long_description_content_type="text/x-rst",
    long_description=readme(),
    url="https://github.com/drjdlarson/gncpy",
    author="Laboratory for Autonomy GNC and Estimation Research (LAGER)",
    author_email="",
    license="MIT",
    packages=find_packages(exclude=["test", "test.*"]),
    install_requires=["numpy", "scipy", "matplotlib"],
    extras_require=extras,
    tests_require=["pytest", "numpy"],
    include_package_data=True,
    ext_modules=[CMakeExtension("gncpy/dynamics/aircraft/lager_super_bindings")],
    python_requires=">=3.7",
    cmdclass=dict(build_ext=CMakeBuild),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
