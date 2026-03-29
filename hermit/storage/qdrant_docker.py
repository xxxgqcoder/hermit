"""Hermit-managed Qdrant Docker container lifecycle.

When QDRANT_MANAGED=true (auto-detected for localhost targets), Hermit
automatically starts a Qdrant Docker container before connecting and
shuts it down on process exit.
"""

import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

# True if Hermit successfully ran `docker run` during this process lifetime.
# Used to guard the atexit cleanup so a failed startup doesn't trigger removal.
_container_created: bool = False


def _is_docker_available() -> bool:
    return shutil.which("docker") is not None


def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    """Poll host:port until it accepts TCP connections or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _wait_for_qdrant_ready(host: str, port: int, timeout: float = 60.0) -> bool:
    """Poll the Qdrant HTTP API until it returns a non-5xx response.

    The TCP port can be open while Qdrant is still initialising inside the
    container (Docker proxy connects immediately).  This function confirms
    the application layer is actually serving requests.
    """
    url = f"http://{host}:{port}/"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _container_exists(name: str) -> bool:
    """Return True if a container with this name exists (any state)."""
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.Name}}", name],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def ensure_qdrant_running(
    host: str,
    port: int,
    grpc_port: int,
    qdrant_data_path: Path,
    container_name: str,
    image: str,
) -> None:
    """Start a fresh Qdrant Docker container. Idempotent at the container level.

    Any existing container with the same name (regardless of state) is removed
    before a new one is created.  This guarantees a clean startup on every
    Hermit launch.

    Raises RuntimeError if Docker is unavailable or the container fails to start.
    """
    global _container_created

    if not _is_docker_available():
        raise RuntimeError(
            "未找到 Docker CLI。Stand-alone 模式需要 Docker，"
            "请安装并启动 Docker Desktop，然后重试。"
        )

    # Remove any pre-existing container with this name (stopped or running)
    if _container_exists(container_name):
        logger.info("Removing existing Qdrant container '%s'...", container_name)
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    qdrant_data_path.mkdir(parents=True, exist_ok=True)
    uid = os.getuid()
    gid = os.getgid()
    logger.info(
        "Creating Qdrant container '%s' from image '%s' "
        "(port %d→6333, %d→6334, data: %s)...",
        container_name, image, port, grpc_port, qdrant_data_path,
    )
    try:
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                "--user", f"{uid}:{gid}",
                "-p", f"{port}:6333",
                "-p", f"{grpc_port}:6334",
                "-v", f"{qdrant_data_path}:/qdrant/storage:z",
                image,
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        raise RuntimeError(
            f"Qdrant 容器 '{container_name}' 启动失败。\n"
            f"错误: {stderr}\n"
            f"提示: 如镜像 '{image}' 不存在，请先运行 `docker pull {image}`，"
            "或通过环境变量 QDRANT_IMAGE 指定正确的镜像版本。"
        ) from e

    _container_created = True
    logger.info("Waiting for Qdrant to become ready at %s:%d (up to 60s)...", host, port)
    if not _wait_for_port(host, port, timeout=60.0):
        raise RuntimeError(
            f"Qdrant 容器 '{container_name}' 启动超时（60秒），请检查 Docker 日志:\n"
            f"  docker logs {container_name}"
        )
    # Port is open but Qdrant may still be initialising — wait for HTTP 200
    if not _wait_for_qdrant_ready(host, port, timeout=60.0):
        raise RuntimeError(
            f"Qdrant 容器 '{container_name}' HTTP API 响应超时（60秒），请检查 Docker 日志:\n"
            f"  docker logs {container_name}"
        )
    logger.info("Qdrant is ready at %s:%d", host, port)


def stop_qdrant_container(container_name: str) -> None:
    """Stop and remove the Hermit-managed Qdrant container.

    No-op if `ensure_qdrant_running` was never successfully called in this
    process (guards against cleanup after a failed startup).
    """
    global _container_created
    if not _container_created:
        return
    logger.info("Removing Qdrant container '%s'...", container_name)
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    _container_created = False
    logger.info("Qdrant container '%s' removed.", container_name)
