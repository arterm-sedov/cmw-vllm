"""CLI interface for cmw-vllm."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from cmw_vllm.health_check import check_server_status, get_server_info, test_inference
from cmw_vllm.logging import setup_logging
from cmw_vllm.model_downloader import check_model_downloaded, download_model
from cmw_vllm.model_registry import get_model_info
from cmw_vllm.model_verifier import verify_model_integrity
from cmw_vllm.server_config import ServerConfig
from cmw_vllm.server_manager import VLLMServerManager

# Setup logging
setup_logging()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """CMW vLLM - vLLM server management tool."""
    if verbose:
        setup_logging("DEBUG")


@cli.command()
def setup() -> None:
    """Initial setup and verification."""
    click.echo("Setting up cmw-vllm...")

    # Check vLLM installation
    try:
        import vllm

        click.echo(f"✓ vLLM installed (version: {vllm.__version__})")
    except ImportError:
        click.echo("✗ vLLM is not installed")
        click.echo("  Install it with: pip install vllm")
        sys.exit(1)

    # Check GPU
    try:
        from cmw_vllm.gpu_info import get_gpu_info

        gpu_info = get_gpu_info()
        if gpu_info["available"]:
            click.echo(f"✓ GPU detected: {gpu_info['name']} ({gpu_info['memory_total_gb']:.2f} GB)")
        else:
            click.echo("⚠ No GPU detected (CPU mode will be used)")
    except Exception as e:
        click.echo(f"⚠ Could not check GPU: {e}")

    # Check HuggingFace
    try:
        from huggingface_hub import __version__

        click.echo(f"✓ huggingface-hub installed (version: {__version__})")
    except ImportError:
        click.echo("✗ huggingface-hub is not installed")
        click.echo("  Install it with: pip install huggingface-hub")
        sys.exit(1)

    click.echo("\n✓ Setup complete!")


@cli.command()
@click.argument("model_id", default="openai/gpt-oss-20b")
@click.option("--local-dir", type=click.Path(path_type=Path), help="Local directory to download to")
@click.option("--no-resume", is_flag=True, help="Don't resume interrupted downloads")
@click.option("--skip-space-check", is_flag=True, help="Skip disk space check")
def download(model_id: str, local_dir: Path | None, no_resume: bool, skip_space_check: bool) -> None:
    """Download model from HuggingFace."""
    click.echo(f"Downloading model: {model_id}")

    # Check if already downloaded
    is_downloaded, model_path = check_model_downloaded(model_id, local_dir)
    if is_downloaded:
        click.echo(f"✓ Model already downloaded at: {model_path}")
        if click.confirm("Re-download?"):
            pass  # Continue with download
        else:
            return

    try:
        downloaded_path = download_model(
            model_id=model_id,
            local_dir=local_dir,
            resume=not no_resume,
            check_space=not skip_space_check,
        )
        click.echo(f"✓ Model downloaded successfully to: {downloaded_path}")

        # Verify
        is_valid, message = verify_model_integrity(downloaded_path, model_id)
        if is_valid:
            click.echo(f"✓ Model verification passed: {message}")
        else:
            click.echo(f"⚠ Model verification warning: {message}")

    except Exception as e:
        click.echo(f"✗ Download failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", help="Model identifier")
@click.option("--port", type=int, help="Server port")
@click.option("--host", help="Server host")
@click.option("--max-model-len", type=int, help="Maximum model length")
@click.option("--gpu-memory-utilization", type=float, help="GPU memory utilization (0.0-1.0)")
@click.option("--cpu-offload-gb", type=int, help="CPU offload memory in GB (for large models on limited GPU memory)")
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (don't detach)")
def start(
    model: str | None,
    port: int | None,
    host: str | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
    cpu_offload_gb: int | None,
    foreground: bool,
) -> None:
    """Start vLLM server."""
    # Create config
    config = ServerConfig.from_env()

    # Override model if provided via CLI
    if model:
        config.model = model
        # Apply model-specific defaults if model changed and settings weren't explicitly provided via CLI
        model_info = get_model_info(config.model)
        if model_info:
            if max_model_len is None and "max_model_len" in model_info:
                config.max_model_len = model_info["max_model_len"]
            if gpu_memory_utilization is None and "gpu_memory_utilization" in model_info:
                config.gpu_memory_utilization = model_info["gpu_memory_utilization"]
            if cpu_offload_gb is None and "cpu_offload_gb" in model_info:
                config.cpu_offload_gb = model_info["cpu_offload_gb"]
    
    # Apply explicit CLI overrides (these take precedence over model defaults)
    if port:
        config.port = port
    if host:
        config.host = host
    if max_model_len is not None:
        config.max_model_len = max_model_len
    if gpu_memory_utilization is not None:
        config.gpu_memory_utilization = gpu_memory_utilization
    if cpu_offload_gb is not None:
        config.cpu_offload_gb = cpu_offload_gb

    click.echo(f"Starting vLLM server with model: {config.model}")
    click.echo(f"Server will be available at: http://{config.host}:{config.port}")

    manager = VLLMServerManager(config)
    
    # Check if server is already running
    if manager.is_running():
        server_info = manager.get_server_info()
        if server_info:
            click.echo(f"⚠ Server is already running (PID: {server_info['pid']})")
            click.echo("  Use 'cmw-vllm stop' to stop it first, or 'cmw-vllm list' to see details")
        else:
            click.echo("⚠ Server appears to be running but could not get details")
        sys.exit(1)
    
    success = manager.start(background=not foreground)

    if success:
        click.echo("✓ Server started successfully")
        if not foreground:
            click.echo(f"  Running in background. Use 'cmw-vllm stop' to stop it.")
    else:
        click.echo("✗ Failed to start server", err=True)
        click.echo("  Check the logs above for error details")
        sys.exit(1)


@cli.command()
def stop() -> None:
    """Stop vLLM server."""
    click.echo("Stopping vLLM server...")

    manager = VLLMServerManager()
    success = manager.stop()

    if success:
        click.echo("✓ Server stopped successfully")
    else:
        click.echo("⚠ Server was not running or failed to stop", err=True)
        sys.exit(1)


@cli.command()
def restart() -> None:
    """Restart vLLM server."""
    click.echo("Restarting vLLM server...")

    manager = VLLMServerManager()
    success = manager.restart()

    if success:
        click.echo("✓ Server restarted successfully")
    else:
        click.echo("✗ Failed to restart server", err=True)
        sys.exit(1)


@cli.command()
@click.option("--all", "-a", "show_all", is_flag=True, help="Show all vLLM servers (not just managed one)")
def list(show_all: bool) -> None:
    """List running vLLM servers."""
    if show_all:
        servers = VLLMServerManager.find_all_servers()
        if not servers:
            click.echo("No vLLM servers are running")
            return

        click.echo(f"Found {len(servers)} running vLLM server(s):")
        click.echo("=" * 80)
        for i, server in enumerate(servers, 1):
            click.echo(f"\nServer {i}:")
            click.echo(f"  PID: {server['pid']}")
            click.echo(f"  Status: {server['status']}")
            click.echo(f"  Memory: {server['memory_mb']:.1f} MB")
            click.echo(f"  Command: {server['cmdline'][:100]}...")
    else:
        manager = VLLMServerManager()
        server_info = manager.get_server_info()
        
        if server_info:
            click.echo("Managed vLLM server is running:")
            click.echo("=" * 80)
            click.echo(f"  PID: {server_info['pid']}")
            click.echo(f"  Status: {server_info['status']}")
            click.echo(f"  Memory: {server_info['memory_mb']:.1f} MB")
            click.echo(f"  CPU: {server_info['cpu_percent']:.1f}%")
            click.echo(f"  Command: {server_info['cmdline'][:100]}...")
        else:
            click.echo("No managed vLLM server is running")
            
            # Check for any vLLM servers
            all_servers = VLLMServerManager.find_all_servers()
            if all_servers:
                click.echo(f"\n⚠ Found {len(all_servers)} unmanaged vLLM server(s) running")
                click.echo("  Use 'cmw-vllm list --all' to see them")


@cli.command()
@click.option("--base-url", default="http://localhost:8000", help="Server base URL")
@click.option("--test-inference", "test_inference_flag", is_flag=True, help="Test inference with a simple request")
def status(base_url: str, test_inference_flag: bool) -> None:
    """Check vLLM server status."""
    click.echo(f"Checking server status at {base_url}...")

    status_info = check_server_status(base_url)

    if status_info["running"]:
        click.echo("✓ Server is running")
        if status_info["models"]:
            click.echo(f"  Available models: {', '.join(status_info['models'])}")
        else:
            click.echo("  No models loaded")

        if test_inference_flag:
            click.echo("\nTesting inference...")
            config = ServerConfig.from_env()
            test_result = test_inference(base_url, config.model)
            if test_result["success"]:
                click.echo(f"✓ Inference test passed")
                click.echo(f"  Response: {test_result['response']}")
            else:
                click.echo(f"✗ Inference test failed: {test_result['error']}")
    else:
        click.echo("✗ Server is not running")
        if status_info["error"]:
            click.echo(f"  Error: {status_info['error']}")


@cli.command()
@click.option("--base-url", default="http://localhost:8000", help="Server base URL")
def info(base_url: str) -> None:
    """Show comprehensive server information."""
    click.echo("Server Information:")
    click.echo("=" * 50)

    server_info = get_server_info()

    # Config
    click.echo("\nConfiguration:")
    config = server_info["config"]
    for key, value in config.items():
        click.echo(f"  {key}: {value}")

    # Status
    click.echo("\nStatus:")
    status = server_info["status"]
    if status["running"]:
        click.echo("  ✓ Running")
        if status["models"]:
            click.echo(f"  Models: {', '.join(status['models'])}")
    else:
        click.echo("  ✗ Not running")
        if status["error"]:
            click.echo(f"  Error: {status['error']}")

    # Inference test
    if "inference_test" in server_info:
        click.echo("\nInference Test:")
        test = server_info["inference_test"]
        if test["success"]:
            click.echo("  ✓ Passed")
            if test["response"]:
                click.echo(f"  Response: {test['response']}")
        else:
            click.echo(f"  ✗ Failed: {test['error']}")


@cli.command()
@click.argument("model_id", default="openai/gpt-oss-20b")
def verify(model_id: str) -> None:
    """Verify model is downloaded and valid."""
    click.echo(f"Verifying model: {model_id}")

    # Check if downloaded
    is_downloaded, model_path = check_model_downloaded(model_id)
    if not is_downloaded:
        click.echo(f"✗ Model not found. Download it with: cmw-vllm download {model_id}")
        sys.exit(1)

    click.echo(f"✓ Model found at: {model_path}")

    # Verify integrity
    is_valid, message = verify_model_integrity(model_path, model_id)
    if is_valid:
        click.echo(f"✓ Model verification passed: {message}")
    else:
        click.echo(f"✗ Model verification failed: {message}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
