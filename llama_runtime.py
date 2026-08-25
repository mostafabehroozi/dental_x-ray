from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "mradermacher/DentalGPT-7B-1026-GGUF"
DEFAULT_MODEL_FILENAME = "DentalGPT-7B-1026.Q6_K.gguf"
DEFAULT_MMPROJ_FILENAME = "DentalGPT-7B-1026.mmproj-f16.gguf"


@dataclass(frozen=True)
class DentalGPTFiles:
    model_path: Path
    mmproj_path: Path


def download_dentalgpt(
    model_dir: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    mmproj_filename: str = DEFAULT_MMPROJ_FILENAME,
    hf_token: str | None = None,
) -> DentalGPTFiles:
    """Download exactly the GGUF language model and matching vision projector."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            local_dir=str(model_dir),
            token=hf_token,
        )
    )
    mmproj_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=mmproj_filename,
            local_dir=str(model_dir),
            token=hf_token,
        )
    )
    return DentalGPTFiles(model_path=model_path, mmproj_path=mmproj_path)


def find_llama_server(explicit_path: str | Path | None = None) -> Path | None:
    """Find a llama-server binary from an explicit path, PATH, or common build paths."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    path_hit = shutil.which("llama-server")
    if path_hit:
        candidates.append(Path(path_hit))

    candidates.extend(
        [
            Path("/kaggle/working/llama.cpp/build/bin/llama-server"),
            Path.cwd() / "llama.cpp" / "build" / "bin" / "llama-server",
        ]
    )

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def build_llama_cpp(
    source_dir: str | Path = "/kaggle/working/llama.cpp",
    cuda_arch: str = "60",
    jobs: int = 4,
    ref: str = "b10516",
    clean_build: bool = False,
) -> Path:
    """
    Build a pinned llama.cpp llama-server with CUDA support.

    Kaggle note:
        Kaggle can provide nvcc/CUDA runtime while CMake fails to expose
        CUDA::cuda_driver.

        GGML_CUDA_NO_VMM=ON avoids that unnecessary direct driver-library
        dependency while keeping CUDA inference enabled.

    P100:
        compute capability = 6.0 -> CMAKE_CUDA_ARCHITECTURES=60
    """

    source_dir = Path(source_dir)
    build_dir = source_dir / "build"
    server = build_dir / "bin" / "llama-server"

    # ---------------------------------------------------------
    # 1. Clone llama.cpp if necessary
    # ---------------------------------------------------------

    if not source_dir.exists():
        print(f"Cloning llama.cpp ref {ref}...")

        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/ggml-org/llama.cpp.git",
                str(source_dir),
            ],
            check=True,
        )

    git_dir = source_dir / ".git"

    if not git_dir.exists():
        raise RuntimeError(
            f"{source_dir} exists but is not a valid llama.cpp git repository."
        )

    # ---------------------------------------------------------
    # 2. Make sure we're actually on the requested pinned ref
    # ---------------------------------------------------------

    print(f"Checking out llama.cpp ref: {ref}")

    subprocess.run(
        [
            "git",
            "-C",
            str(source_dir),
            "fetch",
            "--tags",
            "--force",
        ],
        check=True,
    )

    subprocess.run(
        [
            "git",
            "-C",
            str(source_dir),
            "checkout",
            "--force",
            "--detach",
            ref,
        ],
        check=True,
    )

    commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(source_dir),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    print("llama.cpp commit:", commit)

    # ---------------------------------------------------------
    # 3. Return an already valid binary unless rebuilding
    # ---------------------------------------------------------

    if server.is_file() and not clean_build:
        print("Using existing llama-server:", server)
        return server.resolve()

    # ---------------------------------------------------------
    # 4. Always remove a stale/failed CMake configuration
    # ---------------------------------------------------------

    if build_dir.exists():
        print("Removing previous CMake build directory...")
        shutil.rmtree(build_dir)

    # ---------------------------------------------------------
    # 5. Configure
    # ---------------------------------------------------------

    cmake_command = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),

        # CUDA backend remains enabled.
        "-DGGML_CUDA=ON",

        # Important for Kaggle.
        #
        # Without this, this llama.cpp version tries:
        #
        #   target_link_libraries(
        #       ggml-cuda PRIVATE CUDA::cuda_driver
        #   )
        #
        # Kaggle's CUDA environment may not expose that CMake target.
        "-DGGML_CUDA_NO_VMM=ON",

        # Single GPU notebook: no need for NCCL.
        "-DGGML_CUDA_NCCL=OFF",

        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
    ]

    print("\nConfiguring llama.cpp:")
    print(" ".join(cmake_command))

    subprocess.run(
        cmake_command,
        check=True,
    )

    # ---------------------------------------------------------
    # 6. Compile only llama-server
    # ---------------------------------------------------------

    build_command = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "llama-server",
        "--config",
        "Release",
        "-j",
        str(max(1, jobs)),
    ]

    print("\nBuilding llama-server:")
    print(" ".join(build_command))

    subprocess.run(
        build_command,
        check=True,
    )

    # ---------------------------------------------------------
    # 7. Verify binary
    # ---------------------------------------------------------

    if not server.is_file():
        raise FileNotFoundError(
            "llama-server build finished but binary was not found at:\n"
            f"{server}"
        )

    if not os.access(server, os.X_OK):
        server.chmod(server.stat().st_mode | 0o111)

    print("\nllama-server successfully built:")
    print(server)

    return server.resolve()


class LlamaCppServer:
    """Own a local llama.cpp server process for DentalGPT multimodal inference."""

    def __init__(
        self,
        binary: str | Path,
        model_path: str | Path,
        mmproj_path: str | Path,
        host: str = "127.0.0.1",
        port: int = 8080,
        alias: str = "dentalgpt",
        n_gpu_layers: int = 999,
        ctx_size: int = 8192,
        parallel: int = 1,
        startup_timeout: float = 180.0,
        log_path: str | Path = "/kaggle/working/llama_dentalgpt_server.log",
    ):
        self.binary = Path(binary)
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path)
        self.host = host
        self.port = port
        self.alias = alias
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.parallel = parallel
        self.startup_timeout = startup_timeout
        self.log_path = Path(log_path)
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _healthy(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def start(self, reuse_existing: bool = True) -> None:
        if reuse_existing and self._healthy():
            print(f"Reusing llama.cpp server at {self.base_url}")
            return

        for path in (self.binary, self.model_path, self.mmproj_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")

        command = [
            str(self.binary),
            "--model",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "--alias",
            self.alias,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--n-gpu-layers",
            str(self.n_gpu_layers),
            "--ctx-size",
            str(self.ctx_size),
            "--parallel",
            str(self.parallel),
            # Keep DentalGPT's <think>/<answer> text intact for our parser/logs.
            "--reasoning-format",
            "none",
        ]

        print("Starting llama.cpp DentalGPT server...")
        print(" ".join(command))
        self.process = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                tail = self._tail_log()
                raise RuntimeError(
                    f"llama-server exited with code {self.process.returncode}.\n"
                    f"Last server log lines:\n{tail}"
                )
            if self._healthy():
                print(f"DentalGPT server ready at {self.base_url}")
                return
            time.sleep(1.0)

        self.stop()
        raise TimeoutError(
            f"llama-server did not become healthy within {self.startup_timeout}s.\n"
            f"Last server log lines:\n{self._tail_log()}"
        )

    def _tail_log(self, n: int = 80) -> str:
        if not self.log_path.exists():
            return "<no log file>"
        lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
