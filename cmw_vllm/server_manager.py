"""vLLM server process management."""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from cmw_vllm.server_config import ServerConfig

logger = logging.getLogger(__name__)


class VLLMServerManager:
    """Manages vLLM server process lifecycle."""

    def __init__(self, config: ServerConfig | None = None):
        """Initialize server manager.

        Args:
            config: Server configuration. If None, loads from environment.
        """
        self.config = config or ServerConfig.from_env()
        self.process: subprocess.Popen | None = None
        self.pid_file = Path.home() / ".cmw-vllm" / "server.pid"
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

    def start(self, background: bool = True) -> bool:
        """Start vLLM server.

        Args:
            background: Whether to run in background

        Returns:
            True if server started successfully
        """
        if self.is_running():
            logger.warning("Server is already running")
            return False

        # Check if vLLM is installed
        try:
            import vllm
        except ImportError:
            logger.error("vLLM is not installed. Install it with: pip install vllm")
            return False

        # Build command
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        cmd.extend(self.config.to_vllm_args())

        logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")

        try:
            if background:
                # Start in background
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                # Save PID
                self.pid_file.write_text(str(self.process.pid))
                logger.info(f"Server started in background (PID: {self.process.pid})")
            else:
                # Run in foreground
                subprocess.run(cmd, check=True)
                return True

            # Wait a bit to check if process is still alive
            time.sleep(2)
            if self.process.poll() is not None:
                # Process died
                stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                logger.error(f"Server process exited immediately. Error: {stderr}")
                return False

            logger.info(f"Server started successfully on {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def stop(self) -> bool:
        """Stop vLLM server.

        Returns:
            True if server stopped successfully
        """
        pid = self._get_pid()
        if not pid:
            logger.warning("No server process found")
            return False

        try:
            process = psutil.Process(pid)
            process.terminate()

            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                # Force kill if doesn't terminate
                process.kill()
                process.wait()

            logger.info("Server stopped successfully")
            self._cleanup_pid()
            return True

        except psutil.NoSuchProcess:
            logger.warning("Server process not found")
            self._cleanup_pid()
            return False
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            return False

    def restart(self) -> bool:
        """Restart vLLM server.

        Returns:
            True if server restarted successfully
        """
        logger.info("Restarting server...")
        self.stop()
        time.sleep(2)
        return self.start()

    def is_running(self) -> bool:
        """Check if server is running.

        Returns:
            True if server is running
        """
        pid = self._get_pid()
        if not pid:
            return False

        try:
            process = psutil.Process(pid)
            return process.is_running()
        except psutil.NoSuchProcess:
            self._cleanup_pid()
            return False

    def _get_pid(self) -> int | None:
        """Get server process ID.

        Returns:
            Process ID or None
        """
        if self.process:
            return self.process.pid

        if self.pid_file.exists():
            try:
                return int(self.pid_file.read_text().strip())
            except (ValueError, OSError):
                return None

        return None

    def _cleanup_pid(self) -> None:
        """Clean up PID file."""
        if self.pid_file.exists():
            try:
                self.pid_file.unlink()
            except OSError:
                pass
