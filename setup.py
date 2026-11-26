"""Build gnss_fgo._extra_factors against the installed gtsam shared library."""

import os
import subprocess
import tempfile
from pathlib import Path

from setuptools import Extension, setup


def _find_gtsam_pkg_dir() -> Path:
    env_dir = os.environ.get("GTSAM_LIB_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        import gtsam
        return Path(gtsam.__file__).resolve().parent
    except ImportError:
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            for p in Path(venv).glob("lib/python*/site-packages/gtsam"):
                if (p / "libgtsam.so").exists():
                    return p
        raise RuntimeError(
            "Could not locate gtsam package. Install gtsam first or set GTSAM_LIB_DIR."
        )


def _has_tbb_symbols(lib_path: Path) -> bool:
    if not lib_path.exists():
        return False
    data = lib_path.read_bytes()
    return b"tbb_allocator" in data or b"libtbb" in data


def _ensure_gtsam_headers(gtsam_pkg_dir: Path) -> tuple[Path, Path]:
    src_env = os.environ.get("GTSAM_SRC_DIR")
    bld_env = os.environ.get("GTSAM_BUILD_DIR")
    if src_env and bld_env:
        return Path(src_env), Path(bld_env)

    cache_root = Path(tempfile.gettempdir()) / "gnss_fgo_gtsam_headers"
    bld_dir = cache_root / "build"
    if not (cache_root / "gtsam").is_dir():
        cache_root.parent.mkdir(parents=True, exist_ok=True)
        repo = os.environ.get("GTSAM_GIT_URL", "https://github.com/borglab/gtsam.git")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", repo, str(cache_root)]
        )
    if not (bld_dir / "gtsam" / "config.h").is_file():
        use_tbb = "ON" if _has_tbb_symbols(gtsam_pkg_dir / "libgtsam.so") else "OFF"
        subprocess.check_call([
            "cmake",
            "-S", str(cache_root),
            "-B", str(bld_dir),
            "-DGTSAM_BUILD_PYTHON=OFF",
            "-DGTSAM_BUILD_TESTS=OFF",
            "-DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF",
            "-DGTSAM_USE_SYSTEM_EIGEN=OFF",
            f"-DGTSAM_WITH_TBB={use_tbb}",
        ])
    return cache_root, bld_dir


gtsam_pkg_dir = _find_gtsam_pkg_dir()
gtsam_src, gtsam_bld = _ensure_gtsam_headers(gtsam_pkg_dir)

ext = Extension(
    "gnss_fgo._extra_factors",
    sources=["src/gnss_fgo/cpp/extra_factors.cpp"],
    include_dirs=[
        str(gtsam_src),
        str(gtsam_bld),
        str(gtsam_src / "gtsam" / "3rdparty" / "Eigen"),
        str(gtsam_src / "wrap" / "pybind11" / "include"),
        "/usr/include",
    ],
    library_dirs=[str(gtsam_pkg_dir)],
    libraries=["gtsam"],
    runtime_library_dirs=[str(gtsam_pkg_dir)],
    extra_compile_args=["-O3", "-std=c++17"],
    language="c++",
)

setup(ext_modules=[ext])
