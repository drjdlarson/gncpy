from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.build import check_min_cppstd
from conan.tools.scm.git import Git

required_conan_version = ">=1.53.0"


class GncpyConan(ConanFile):
    name = "gncpy"
    version = "0.1.0"

    # Optional metadata
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of Gncpy here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    # Sources are located in the same place as this recipe, copy them to the recipe
    # exports_sources = "CMakeLists.txt", "src/*", "include/*"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        cmake_layout(self)
    
    def validate(self):
        if self.settings.get_safe("compiler.cppstd"):
            check_min_cppstd(self, 20)

    def source(self):   
        # TODO: temporary until there are full releases on conan center (should this be controlled by a setting?)
        git = Git(self, folder=".")
        git.clone("https://github.com/ryan4984/gncpy_cpp", target=".")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["GNCPY_DOC"] = False
        tc.cache_variables["GNCPY_TEST"] = False
        tc.cache_variables["GNCPY_INSTALL"] = True
        tc.cache_variables["GNCPY_LIB_DIR"] = "lib"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        target = "gncpy"
        self.cpp_info.set_property("cmake_file_name", "gncpy")
        self.cpp_info.set_property("cmake_target_name", f"lager::{target}")
        self.cpp_info.set_property("pkg_config_name",  "gncpy")

        # TODO: back to global scope in conan v2 once cmake_find_package* generators removed
        postfix = "d" if self.settings.build_type == "Debug" else ""
        libname = "gncpy" + postfix
        self.cpp_info.components["_gncpy"].libs = [libname]
        if self.settings.os == "Linux":
            self.cpp_info.components["_gncpy"].system_libs.extend(["m"])

        # TODO: to remove in conan v2 once cmake_find_package* generators removed
        self.cpp_info.names["cmake_find_package"] = "gncpy"
        self.cpp_info.names["cmake_find_package_multi"] = "gncpy"
        self.cpp_info.names["pkg_config"] = "gncpy"
        self.cpp_info.components["_gncpy"].names["cmake_find_package"] = target
        self.cpp_info.components["_gncpy"].names["cmake_find_package_multi"] = target
        self.cpp_info.components["_gncpy"].set_property("cmake_target_name", f"lager::{target}")
