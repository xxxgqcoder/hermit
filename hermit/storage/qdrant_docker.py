"""Hermit-managed Qdrant Docker container lifecycle.

When QDRANT_MANAGED=true (auto-detected for localhost targets), Hermit
automatically ensures a Qdrant Docker container is running before connecting
and *stops* it (without removing) on process exit.

Persistent-container design
────────────────────────────
The Qdrant container is treated as persistent storage infrastructure whose
lifetime is independent of the hermit process:

  • hermit start  — creates the container on first run; on subsequent runs
                    it either adopts a live container (fast-path) or restarts
                    a stopped one with `docker start`.
  • hermit stop   — issues `docker stop` (not rm -f).  The container survives
                    as "stopped", consuming no CPU/memory but retaining its
                    data volume and port config.
  • crash/SIGKILL — the container keeps running; next `hermit start` adopts
                    it via the fast-path healthy check.

To force-remove the container (e.g. for a clean reset) use:
    docker rm -f hermit_qdrant
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

# True when this process has taken ownership of the container (either by
# creating it, restarting a stopped instance, or adopting a running one).
# Guards atexit so a failed pre-startup state doesn't trigger stop.
_container_managed: bool = False


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


def _is_image_present(image: str) -> bool:
    """Return True if the Docker image exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def _pull_image(image: str) -> None:
    """Pull a Docker image, streaming progress to the log.

    Runs docker pull without capturing output so that progress lines are
    visible in the server log file, giving operators visibility into the
    pull duration.

    Raises RuntimeError on failure.
    """
    logger.info("镜像 '%s' 本地不存在，开始拉取（此步骤可能需要数分钟，取决于网络速度）...", image)
    result = subprocess.run(
        ["docker", "pull", image],
        capture_output=False,  # stream progress to server log
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Docker 镜像 '{image}' 拉取失败（退出码 {result.returncode}）。\n"
            f"请检查网络连接，或通过环境变量 QDRANT_IMAGE 指定正确的镜像版本。"
        )
    logger.info("镜像 '%s' 拉取完成。", image)


def ensure_qdrant_running(
    host: str,
    port: int,
    grpc_port: int,
    qdrant_data_path: Path,
    container_name: str,
    image: str,
) -> None:
    """Ensure a Qdrant Docker container is running and healthy.  Idempotent.

    Priority order:
      1. Adopt — Qdrant already healthy on the expected port (fast-path, no
                 Docker commands needed).  This handles the crash-and-restart
                 scenario where the container kept running after hermit died.
      2. Restart — Container exists but is stopped; attempt `docker start`.
                   Falls back to recreate if start fails or port doesn't
                   become healthy (e.g. port mapping mismatch).
      3. Create  — No container with this name; run a new one.

    Raises RuntimeError if Docker is unavailable or the container fails to start.
    """
    global _container_managed

    if not _is_docker_available():
        raise RuntimeError(
            "未找到 Docker CLI。Stand-alone 模式需要 Docker，"
            "请安装并启动 Docker Desktop，然后重试。"
        )

    # ── 1. Adopt: already healthy ────────────────────────────────
    if _wait_for_qdrant_ready(host, port, timeout=2.0):
        logger.info(
            "Qdrant already healthy at %s:%d — adopting existing container.",
            host, port,
        )
        _container_managed = True
        return

    # ── 2. Restart: stopped container ───────────────────────────
    if _container_exists(container_name):
        logger.info("Found stopped container '%s', attempting docker start...", container_name)
        result = subprocess.run(
            ["docker", "start", container_name],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and _wait_for_qdrant_ready(host, port, timeout=30.0):
            logger.info("Restarted container '%s' successfully.", container_name)
            _container_managed = True
            return
        # Start failed or container came up on wrong port — recreate
        logger.warning(
            "Container '%s' restart failed or unhealthy on port %d — recreating.",
            container_name, port,
        )
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    # ── 3. Create: fresh container ───────────────────────────────
    qdrant_data_path.mkdir(parents=True, exist_ok=True)

    # Ensure the image is available locally before running the container.
    # docker run silently pulls missing images with no progress output —
    # on a fresh machine this can take minutes with zero feedback.
    if not _is_image_present(image):
        _pull_image(image)  # raises RuntimeError on failure

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
                "--user", f"{os.getuid()}:{os.getgid()}",
                "-p", f"{port}:6333",
                "-p", f"{grpc_port}:6334",
                "-v", f"{qdrant_data_path}:/qdrant/storage:z",
                "-e", "QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/storage/snapshots",
                image,
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        raise RuntimeError(
            f"Qdrant 容器 '{container_name}' 启动失败。\n"
            f"错误: {stderr}\n"
            f"提示: 通过环境变量 QDRANT_IMAGE 指定正确的镜像版本，"
            f"或手动运行 `docker pull {image}` 后重试。"
        ) from e

    _container_managed = True
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
    """Stop (but do NOT remove) the Hermit-managed Qdrant container.

    Persistent-container design: the container is stopped rather than deleted
    so that subsequent hermit starts can resume quickly via `docker start`.
    Data is preserved on the host-mounted volume regardless.

    No-op if this process never successfully managed a container (guards
    against atexit cleanup after a failed startup).
    """
    global _container_managed
    if not _container_managed:
        return
    logger.info("Stopping Qdrant container '%s'...", container_name)
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    _container_managed = False
    logger.info("Qdrant container '%s' stopped (data preserved).", container_name)


def remove_qdrant_container(container_name: str) -> None:
    """Force-remove the Qdrant container and release its name.

    Use this for a clean reset or in test teardown.  Data on the host
    volume is NOT affected (volume is managed separately).
    """
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
