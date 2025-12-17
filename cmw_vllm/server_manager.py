"""vLLM server process management."""
from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import threading
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

        # Build command - use wrapper script to register plugin before vLLM validates parsers
        wrapper_script = str(Path(__file__).parent / "vllm_wrapper.py")
        project_root = Path(__file__).parent.parent
        cmd = [sys.executable, wrapper_script]
        # Set PYTHONPATH to include project root so cmw_vllm can be imported
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        if pythonpath:
            env["PYTHONPATH"] = f"{project_root}:{pythonpath}"
        else:
            env["PYTHONPATH"] = str(project_root)
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
                    env=env,
                )
                # Save PID
                self.pid_file.write_text(str(self.process.pid))
                logger.info(f"Server started in background (PID: {self.process.pid})")
            else:
                # Run in foreground
                subprocess.run(cmd, check=True, env=env)
                return True

            # Capture stderr in background to see errors
            stderr_lines = []
            stdout_lines = []
            
            def capture_output(pipe, lines_list):
                try:
                    for line in iter(pipe.readline, b''):
                        if line:
                            lines_list.append(line.decode('utf-8', errors='replace').strip())
                except Exception:
                    pass
            
            if self.process.stderr:
                stderr_thread = threading.Thread(
                    target=capture_output,
                    args=(self.process.stderr, stderr_lines),
                    daemon=True
                )
                stderr_thread.start()
            
            if self.process.stdout:
                stdout_thread = threading.Thread(
                    target=capture_output,
                    args=(self.process.stdout, stdout_lines),
                    daemon=True
                )
                stdout_thread.start()
            
            # Wait and check if process is still alive, and if port is listening
            max_wait = 30  # Wait up to 30 seconds
            check_interval = 1  # Check every second
            waited = 0
            
            while waited < max_wait:
                time.sleep(check_interval)
                waited += check_interval
                
                # Check if process died
                if self.process.poll() is not None:
                    # Process died - get error output
                    time.sleep(0.5)  # Give threads time to capture output
                    error_output = '\n'.join(stderr_lines[-20:] + stdout_lines[-20:])  # Last 20 lines of each
                    if error_output:
                        logger.error(f"Server process exited. Last output:\n{error_output}")
                    else:
                        logger.error("Server process exited immediately with no output")
                    self._cleanup_pid()
                    return False
                
                # Check if port is listening
                if self._is_port_listening():
                    logger.info(f"Server started successfully on {self.config.host}:{self.config.port}")
                    return True
                
                # Log progress for long waits
                if waited % 5 == 0:
                    logger.info(f"Waiting for server to start... ({waited}s)")
            
            # Timeout - check if process is still alive
            if self.process.poll() is None:
                # Process is alive but port not listening - might still be loading
                logger.warning(f"Server process is running but port {self.config.port} not yet listening after {max_wait}s")
                logger.warning("Server may still be loading the model. Check status with 'cmw-vllm list'")
                return True  # Consider it started if process is alive
            else:
                # Process died during wait
                time.sleep(0.5)
                error_output = '\n'.join(stderr_lines[-20:] + stdout_lines[-20:])
                if error_output:
                    logger.error(f"Server process exited. Last output:\n{error_output}")
                else:
                    logger.error("Server process exited with no output")
                self._cleanup_pid()
                return False

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
            # Check if process is actually running and is a vLLM server
            if not process.is_running():
                self._cleanup_pid()
                return False
            
            # Verify it's actually a vLLM server process
            cmdline_str = ' '.join(process.cmdline())
            if 'vllm' not in cmdline_str.lower() and 'api_server' not in cmdline_str.lower():
                # PID exists but it's not our vLLM server
                self._cleanup_pid()
                return False
                
            return True
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
        self.process = None

    def get_server_info(self) -> dict | None:
        """Get information about the running server.

        Returns:
            Dictionary with server info or None if not running
        """
        pid = self._get_pid()
        if not pid:
            return None

        try:
            process = psutil.Process(pid)
            if not process.is_running():
                self._cleanup_pid()
                return None

            cmdline = ' '.join(process.cmdline())
            if 'vllm' not in cmdline.lower() and 'api_server' not in cmdline.lower():
                self._cleanup_pid()
                return None

            return {
                "pid": pid,
                "cmdline": cmdline,
                "status": process.status(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(interval=0.1),
                "create_time": process.create_time(),
            }
        except psutil.NoSuchProcess:
            self._cleanup_pid()
            return None

    def _is_port_listening(self) -> bool:
        """Check if the server port is listening.
        
        Returns:
            True if port is listening
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.config.host if self.config.host != '0.0.0.0' else 'localhost', self.config.port))
            sock.close()
            return result == 0
        except Exception:
            return False

    @staticmethod
    def find_all_servers() -> list[dict]:
        """Find all running vLLM server processes.

        Returns:
            List of dictionaries with server information
        """
        servers = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status', 'memory_info', 'create_time']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'vllm.entrypoints.openai.api_server' in cmdline or 'api_server' in cmdline:
                    servers.append({
                        "pid": proc.info['pid'],
                        "cmdline": cmdline,
                        "status": proc.info['status'],
                        "memory_mb": proc.info['memory_info'].rss / 1024 / 1024 if proc.info['memory_info'] else 0,
                        "create_time": proc.info['create_time'],
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return servers
