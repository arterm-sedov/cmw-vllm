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
@click.argument("model_id", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
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
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (don't detach)")
def start(
    model: str | None,
    port: int | None,
    host: str | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
    foreground: bool,
) -> None:
    """Start vLLM server."""
    # Create config
    config = ServerConfig.from_env()

    # Override with CLI options
    if model:
        config.model = model
    if port:
        config.port = port
    if host:
        config.host = host
    if max_model_len:
        config.max_model_len = max_model_len
    if gpu_memory_utilization is not None:
        config.gpu_memory_utilization = gpu_memory_utilization

    click.echo(f"Starting vLLM server with model: {config.model}")
    click.echo(f"Server will be available at: http://{config.host}:{config.port}")

    manager = VLLMServerManager(config)
    success = manager.start(background=not foreground)

    if success:
        click.echo("✓ Server started successfully")
        if not foreground:
            click.echo(f"  Running in background. Use 'cmw-vllm stop' to stop it.")
    else:
        click.echo("✗ Failed to start server", err=True)
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
@click.option("--base-url", default="http://localhost:8000", help="Server base URL")
@click.option("--test-inference", is_flag=True, help="Test inference with a simple request")
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
@click.argument("model_id", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
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
