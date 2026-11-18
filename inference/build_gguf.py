#!/usr/bin/env python3
"""Build the CPU GGUF inference library from the pinned llama.cpp revision."""

import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from checkpoint import LLAMA_CPP_COMMIT

LLAMA_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_CACHE = Path.home() / ".cache/wepm-gguf"


def source_revision(source):
    # The offline converter's source archive includes this identity marker.
    marker = source / "SOURCE_COMMIT"
    if marker.is_file() and not (source / ".git").exists():
        return marker.read_text().strip()
    return subprocess.check_output(
        ["git", "-C", source, "rev-parse", "HEAD"], text=True
    ).strip()


def fetch_source(path):
    if not path.exists():
        with TemporaryDirectory(dir=path.parent, prefix=".llama-fetch-") as tmp:
            checkout = Path(tmp) / "llama.cpp"
            subprocess.run(["git", "init", "-q", checkout], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    checkout,
                    "fetch",
                    "--depth=1",
                    LLAMA_REPOSITORY,
                    LLAMA_CPP_COMMIT,
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", checkout, "checkout", "--detach", "-q", LLAMA_CPP_COMMIT],
                check=True,
            )
            checkout.rename(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="parallel build jobs")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--llama-source", type=Path, help="existing pinned llama.cpp source"
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not shutil.which("cmake"):
        parser.error("CMake 3.20 or later is required")
    cache = args.cache.expanduser().resolve() / LLAMA_CPP_COMMIT
    cache.mkdir(parents=True, exist_ok=True)
    source = (
        args.llama_source.expanduser().resolve()
        if args.llama_source
        else fetch_source(cache / "llama.cpp")
    )
    if source_revision(source) != LLAMA_CPP_COMMIT:
        parser.error(f"llama.cpp source must match {LLAMA_CPP_COMMIT}")
    build = cache / "native-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            Path(__file__).resolve().parent / "native",
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DLLAMA_CPP_ROOT={source}",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            build,
            "--target",
            "wepm_gguf",
            "--parallel",
            str(args.threads),
        ],
        check=True,
    )
    print(f"WePM GGUF runtime is in {build / 'bin'}")


if __name__ == "__main__":
    main()
