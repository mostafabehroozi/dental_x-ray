from __future__ import annotations

import hashlib
import json
import re
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, snapshot_download


# DentVLM is published only as bf16 safetensors (Hugging Face ZJU-AI4H/DentVLM, gated with
# automatic approval, CC BY-NC 4.0). convert_to_gguf() turns it into the two GGUF files below
# once; keep them in a private Kaggle dataset or Hugging Face repo and point the notebook at it.
DENTVLM_HF_REPO_ID = "ZJU-AI4H/DentVLM"
DENTVLM_REVISION = "2ad8e71ea6708eee92723e7eca6e30e6dac48d85"
RUNTIME_REF = "b10516"
PROVENANCE_FILE = "dentvlm_provenance.json"

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verified_model_provenance(files):
    path = files.model_path.parent / PROVENANCE_FILE
    if not path.is_file():
        raise ValueError(f"{path} missing: use a fresh model directory for pinned conversion, or copy its provenance with both GGUF files")
    data = json.loads(path.read_text(encoding="utf8"))
    if (data.get("hf_repo") != DENTVLM_HF_REPO_ID or data.get("hf_revision") != DENTVLM_REVISION
            or data.get("outtype") != "q8_0"):
        raise ValueError("Model provenance does not match the pinned Q8 DentVLM checkpoint")
    for key, file in (("model_sha256", files.model_path), ("mmproj_sha256", files.mmproj_path)):
        if data.get(key) != sha256_file(file):
            raise ValueError(f"{file}: hash does not match conversion provenance")
    if not data.get("converter_revision") or not data.get("checkpoint_template_sha256"):
        raise ValueError("Conversion provenance lacks converter or checkpoint chat-template identity")
    return data

DEFAULT_MODEL_FILENAME = "DentVLM-Q8_0.gguf"
DEFAULT_MMPROJ_FILENAME = "DentVLM-mmproj-f16.gguf"


@dataclass(frozen=True)
class ModelFiles:
    model_path: Path
    mmproj_path: Path


def download_gguf(
    model_dir: str | Path,
    repo_id: str,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    mmproj_filename: str = DEFAULT_MMPROJ_FILENAME,
    hf_token: str | None = None,
    revision: str | None = None,
) -> ModelFiles:
    """Download exactly the GGUF language model and matching vision projector."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("GGUF download requires a pinned commit revision")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(hf_hub_download(repo_id=repo_id, filename=model_filename, local_dir=str(model_dir), token=hf_token, revision=revision))
    mmproj_path = Path(hf_hub_download(repo_id=repo_id, filename=mmproj_filename, local_dir=str(model_dir), token=hf_token, revision=revision))
    hf_hub_download(repo_id=repo_id, filename=PROVENANCE_FILE, local_dir=str(model_dir), token=hf_token, revision=revision)
    return local_gguf(model_dir, model_filename, mmproj_filename)


def local_gguf(model_dir: str | Path, model_filename: str = DEFAULT_MODEL_FILENAME,
               mmproj_filename: str = DEFAULT_MMPROJ_FILENAME) -> ModelFiles:
    """Use GGUF files already on disk (for example an attached Kaggle dataset)."""
    model_dir = Path(model_dir)
    files = ModelFiles(model_path=model_dir / model_filename, mmproj_path=model_dir / mmproj_filename)
    for path in (files.model_path, files.mmproj_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    verified_model_provenance(files)
    return files


def convert_to_gguf(
    out_dir: str | Path,
    llama_cpp_dir: str | Path,
    hf_repo_id: str = DENTVLM_HF_REPO_ID,
    hf_token: str | None = None,
    work_dir: str | Path = "/tmp/dentvlm_hf",
    outtype: str = "q8_0",
    model_filename: str = DEFAULT_MODEL_FILENAME,
    mmproj_filename: str = DEFAULT_MMPROJ_FILENAME,
    install_requirements: bool = True,
    revision: str = DENTVLM_REVISION,
) -> ModelFiles:
    """One-time conversion of the Hugging Face checkpoint to GGUF (language model + mmproj).

    Needs the gated repo accepted on the Hugging Face model page and a token, about 17 GB of
    scratch disk for the safetensors under work_dir, and llama.cpp's converter requirements.
    The converter writes the language model directly at the requested outtype (q8_0 by
    default) and the vision projector in f16, so llama-quantize is not needed.
    """
    if revision != DENTVLM_REVISION or hf_repo_id != DENTVLM_HF_REPO_ID or outtype != "q8_0":
        raise ValueError("This profile requires the pinned DentVLM checkpoint and Q8 conversion")
    out_dir, llama_cpp_dir, work_dir = Path(out_dir), Path(llama_cpp_dir), Path(work_dir)
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"{converter} (build_llama_cpp clones the repository)")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = ModelFiles(model_path=out_dir / model_filename, mmproj_path=out_dir / mmproj_filename)
    if files.model_path.is_file() and files.mmproj_path.is_file():
        verified_model_provenance(files)
        print("Verified GGUF files already present:", out_dir)
        return files
    if files.model_path.exists() or files.mmproj_path.exists():
        raise ValueError("Partial unverified conversion: use a fresh model directory")

    if install_requirements:
        requirements = llama_cpp_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=True)

    print(f"Downloading {hf_repo_id} to {work_dir} ...")
    hf_dir = Path(snapshot_download(repo_id=hf_repo_id, local_dir=str(work_dir), token=hf_token, revision=revision,
                                    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))

    if not files.mmproj_path.is_file():
        print("Converting the vision projector (f16) ...")
        subprocess.run([sys.executable, str(converter), str(hf_dir), "--mmproj", "--outtype", "f16",
                        "--outfile", str(files.mmproj_path)], check=True)
    if not files.model_path.is_file():
        print(f"Converting the language model ({outtype}) ...")
        subprocess.run([sys.executable, str(converter), str(hf_dir), "--outtype", outtype,
                        "--outfile", str(files.model_path)], check=True)
    tokenizer = json.loads((hf_dir / "tokenizer_config.json").read_text(encoding="utf8"))
    template = tokenizer.get("chat_template")
    if not template:
        raise ValueError("Pinned checkpoint has no chat template to verify")
    converter_revision = subprocess.check_output(["git", "-C", str(llama_cpp_dir), "rev-parse", "HEAD"], text=True).strip()
    provenance = {"hf_repo": hf_repo_id, "hf_revision": revision, "outtype": outtype,
                  "converter_revision": converter_revision, "converter_sha256": sha256_file(converter),
                  "checkpoint_template_sha256": hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest(),
                  "model_sha256": sha256_file(files.model_path), "mmproj_sha256": sha256_file(files.mmproj_path),
                  "native_runtime_equivalence": "unverified_quantized_approximation"}
    (out_dir / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2), encoding="utf8")
    for path in (files.model_path, files.mmproj_path):
        print(f"{path} ({path.stat().st_size / 1024**3:.2f} GiB)")
    return files


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

    stamp = build_dir / ".commit"
    built = stamp.read_text().strip() if stamp.is_file() else None
    if server.is_file() and not clean_build and built == commit:
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
    stamp.write_text(commit)

    print("\nllama-server successfully built:")
    print(server)

    return server.resolve()


class LlamaCppServer:
    """Own a local llama.cpp server process for DentVLM multimodal inference."""

    def __init__(
        self,
        binary: str | Path,
        model_path: str | Path,
        mmproj_path: str | Path,
        host: str = "127.0.0.1",
        port: int = 8080,
        alias: str = "dentvlm",
        n_gpu_layers: int = 999,
        ctx_size: int = 16384,
        parallel: int = 1,
        image_max_tokens: int | None = 8192,
        image_min_tokens: int | None = 4,
        startup_timeout: float = 180.0,
        log_path: str | Path = "/kaggle/working/llama_dentvlm_server.log",
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
        # The authors run DentVLM with max_pixels = 8192 x 28 x 28, i.e. up to 8192 image
        # tokens, and no floor beyond 4 tokens; llama.cpp would otherwise cap Qwen2-VL images
        # at 4096 tokens and downscale a full-size panoramic. ctx must hold image + prompt +
        # answer (the authors' max input length is 16384).
        self.image_max_tokens = image_max_tokens
        self.image_min_tokens = image_min_tokens
        self.startup_timeout = startup_timeout
        self.log_path = Path(log_path)
        self.process: subprocess.Popen | None = None
        self._log_handle = None
        self.provenance = {}
        self.identity_path = self.log_path.with_suffix(".identity.json")
        if (ctx_size, parallel, image_min_tokens, image_max_tokens) != (16384, 1, 4, 8192):
            raise ValueError("PAN runtime requires context 16384, one slot, and image bounds 4..8192")

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
        if self._healthy():
            if not reuse_existing:
                raise RuntimeError("Port already has a server; stop it or explicitly request verified reuse")
            self.verify_runtime(reusing=True)
            print(f"Reusing verified llama.cpp server at {self.base_url}")
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
        ]
        command += ["--temp", "0.1", "--top-p", "0.001", "--top-k", "0", "--min-p", "0",
                    "--repeat-penalty", "1.05", "--repeat-last-n", "-1", "--seed", "0",
                    "--samplers", "penalties;temperature;top_p", "--n-predict", "512"]
        if self.image_max_tokens:
            command += ["--image-max-tokens", str(self.image_max_tokens)]
        if self.image_min_tokens:
            command += ["--image-min-tokens", str(self.image_min_tokens)]

        self.provenance = self._identity()
        self.provenance["launch_command"] = command
        print("Starting llama.cpp DentVLM server...")
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
                try:
                    self.verify_runtime()
                except Exception:
                    self.stop()
                    raise
                self.identity_path.write_text(json.dumps({"pid": self.process.pid, "provenance": self.provenance}, indent=2), encoding="utf8")
                print(f"DentVLM server ready and verified at {self.base_url}")
                return
            time.sleep(1.0)

        self.stop()
        raise TimeoutError(
            f"llama-server did not become healthy within {self.startup_timeout}s.\n"
            f"Last server log lines:\n{self._tail_log()}"
        )

    def _identity(self):
        model = verified_model_provenance(ModelFiles(self.model_path, self.mmproj_path))
        version = subprocess.check_output([str(self.binary), "--version"], text=True, stderr=subprocess.STDOUT)
        if not re.search(r"\b10516\b", version):
            raise ValueError("llama-server must be built from pinned b10516")
        return {"conversion": model, "binary_sha256": sha256_file(self.binary), "runtime_version": version.strip(),
                "model_path": str(self.model_path.resolve()), "mmproj_path": str(self.mmproj_path.resolve()),
                "host": self.host, "port": self.port, "ctx_size": self.ctx_size, "parallel": self.parallel, "image_min_tokens": self.image_min_tokens,
                "image_max_tokens": self.image_max_tokens, "n_gpu_layers": self.n_gpu_layers, "alias": self.alias}

    def verify_runtime(self, reusing=False):
        """Fail closed before inference; record real template and server sampling properties."""
        from dental_pipeline import SYSTEM_MESSAGE, TASKS
        if self.process is not None and self.process.poll() is not None:
            raise ValueError("The verified server process has stopped")
        if reusing:
            if not self.identity_path.is_file():
                raise ValueError("Cannot reuse an unidentified server")
            saved = json.loads(self.identity_path.read_text(encoding="utf8"))
            identity = self._identity()
            if any(saved["provenance"].get(k) != v for k, v in identity.items()):
                raise ValueError("Existing server identity/settings do not match")
            # A stale identity file must not bless another process on the same port.
            command_path = Path(f"/proc/{saved['pid']}/cmdline")
            if not command_path.is_file():
                raise ValueError("Cannot verify the existing server process; restart it from this notebook")
            args = command_path.read_bytes().decode().rstrip("\0").split("\0")
            if args != saved["provenance"].get("launch_command"):
                raise ValueError("Existing server process command differs from its identity record")
            self.provenance = saved["provenance"]
        if not self.provenance:
            raise ValueError("Start and verify the pinned server before creating an inference runner")
        response = requests.get(f"{self.base_url}/props", timeout=30)
        response.raise_for_status()
        props = response.json()
        defaults = props.get("default_generation_settings", {})
        params = defaults.get("params", {})
        expected = {"temperature": 0.1, "top_p": 0.001, "repeat_penalty": 1.05,
                    "repeat_last_n": -1, "seed": 0, "top_k": 0, "min_p": 0}
        for key, value in expected.items():
            actual = params.get(key)
            if key == "repeat_last_n" and actual == self.ctx_size:
                continue  # llama.cpp may resolve -1 to the full context length
            if not isinstance(actual, (int, float)) or abs(actual - value) > 1e-6:
                raise ValueError(f"Effective {key}={actual!r}; expected {value}")
        if params.get("samplers") != ["penalties", "temperature", "top_p"]:
            raise ValueError("Effective sampler order differs from penalties -> temperature -> top_p")
        if defaults.get("n_ctx") != self.ctx_size or props.get("total_slots") != 1:
            raise ValueError("Effective context/slot settings differ from the PAN contract")
        # The image bounds are process-level flags (not exposed by /props). Verify the actual
        # owned/reused process arguments rather than treating an arbitrary healthy server as proof.
        command = self.provenance.get("launch_command", [])
        for flag, value in (("--image-min-tokens", "4"), ("--image-max-tokens", "8192"), ("--n-predict", "512")):
            if flag not in command or command[command.index(flag) + 1] != value:
                raise ValueError("Image/output token limits were not established by the verified process")
        if Path(props.get("model_path", "")).resolve() != self.model_path.resolve() or props.get("model_alias") != self.alias:
            raise ValueError("Server reports a different model path or alias")
        if props.get("modalities", {}).get("vision") is not True:
            raise ValueError("Server has no verified vision input")
        media_marker = props.get("media_marker")
        if not isinstance(media_marker, str) or not media_marker:
            raise ValueError("Server does not expose its multimodal marker")
        rendered = {}
        marker = "<|vision_start|><|image_pad|><|vision_end|>"
        for task, block in TASKS.items():
            question = block["questions"][0]
            # Probe the actual image_url + text shape used by VisionRunner, without generation.
            # The native image embedding expansion occurs later inside mtmd.
            png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS1sAAAAASUVORK5CYII="
            messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": png}},
                                                      {"type": "text", "text": question}]}]
            response = requests.post(f"{self.base_url}/apply-template", json={"messages": messages, "add_generation_prompt": True}, timeout=30)
            response.raise_for_status()
            prompt = response.json().get("prompt")
            expected_prompt = f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n{marker}{question}<|im_end|>\n<|im_start|>assistant\n"
            if not isinstance(prompt, str) or prompt.count(media_marker) != 1:
                raise ValueError("Multimodal chat wrapper did not preserve exactly one image input")
            wrapped = "<|vision_start|>" + media_marker + "<|vision_end|>"
            normalized = prompt.replace(wrapped, marker) if wrapped in prompt else prompt.replace(media_marker, marker)
            if normalized != expected_prompt:
                raise ValueError(f"Chat wrapper mismatch for {task}; inspect /apply-template before inference")
            rendered[task] = normalized
        self.provenance.update(effective_generation={**expected, "samplers": params["samplers"], "max_tokens": 512},
                               server_generation_properties=params, rendered_templates=rendered,
                               output_limit_verification="request max_tokens=512 and process --n-predict=512; props task default is not the process cap",
                               chat_template=props.get("chat_template"),
                               template_verification="multimodal apply-template; image embedding marker normalized",
                               image_limits_verification="verified_process_arguments; native preprocessing equivalence unverified")
        return self.provenance

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
