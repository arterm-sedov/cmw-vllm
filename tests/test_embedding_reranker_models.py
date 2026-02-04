"""Test model registry for new embedding and reranking models."""

import pytest

from cmw_vllm.model_registry import MODEL_REGISTRY, get_model_info


class TestEmbeddingModels:
    """Tests for embedding model configurations."""

    def test_qwen3_embedding_model(self):
        """Test Qwen3-Embedding-0.6B model configuration."""
        model_id = "Qwen/Qwen3-Embedding-0.6B"
        model_info = get_model_info(model_id)

        assert model_info is not None, f"Model {model_id} not found in registry"
        assert model_info["task"] == "embed", "Task should be 'embed' for embedding models"
        assert model_info["runner"] == "pooling", "Runner should be 'pooling' for embedding models"
        assert model_info["dtype"] == "float16", "Dtype should be float16"
        assert model_info["max_model_len"] == 32768, "Max model len should be 32768"
        assert model_info["size_gb"] > 0, "Size should be positive"
        assert "description" in model_info, "Should have description"

    def test_frida_model(self):
        """Test ai-forever/FRIDA model configuration."""
        model_id = "ai-forever/FRIDA"
        model_info = get_model_info(model_id)

        assert model_info is not None, f"Model {model_id} not found in registry"
        assert model_info["task"] == "embed", "Task should be 'embed' for embedding models"
        assert model_info["runner"] == "pooling", "Runner should be 'pooling' for embedding models"
        assert model_info["dtype"] == "float16", "Dtype should be float16"
        assert model_info["max_model_len"] == 512, "FRIDA should have 512 max len"
        assert model_info["gpu_memory_utilization"] == 0.3, "Should use lower GPU memory for small models"


class TestRerankerModels:
    """Tests for reranker model configurations."""

    def test_qwen3_reranker_model(self):
        """Test Qwen3-Reranker-0.6B model configuration."""
        model_id = "Qwen/Qwen3-Reranker-0.6B"
        model_info = get_model_info(model_id)

        assert model_info is not None, f"Model {model_id} not found in registry"
        assert model_info["task"] == "score", "Task should be 'score' for reranking models"
        assert model_info["runner"] == "pooling", "Runner should be 'pooling' for reranking models"
        assert model_info["dtype"] == "float16", "Dtype should be float16"
        assert model_info["max_model_len"] == 32768, "Max model len should be 32768"
        assert "description" in model_info, "Should have description"

    def test_bge_reranker_model(self):
        """Test BAAI/bge-reranker-v2-m3 model configuration."""
        model_id = "BAAI/bge-reranker-v2-m3"
        model_info = get_model_info(model_id)

        assert model_info is not None, f"Model {model_id} not found in registry"
        assert model_info["task"] == "score", "Task should be 'score' for reranking models"
        assert model_info["runner"] == "pooling", "Runner should be 'pooling' for reranking models"
        assert model_info["dtype"] == "float16", "Dtype should be float16"
        assert model_info["max_model_len"] == 8192, "BGE reranker supports 8K context"
        assert "hf_overrides" in model_info, "Should have hf_overrides for BGE-M3"
        assert "BgeM3EmbeddingModel" in model_info["hf_overrides"], "Should specify BGE-M3 architecture"

    def test_russian_reranker_model(self):
        """Test DiTy/cross-encoder-russian-msmarco model configuration."""
        model_id = "DiTy/cross-encoder-russian-msmarco"
        model_info = get_model_info(model_id)

        assert model_info is not None, f"Model {model_id} not found in registry"
        assert model_info["task"] == "score", "Task should be 'score' for reranking models"
        assert model_info["runner"] == "pooling", "Runner should be 'pooling' for reranking models"
        assert model_info["dtype"] == "float16", "Dtype should be float16"
        assert model_info["max_model_len"] == 512, "Russian reranker should have 512 max len"
        assert model_info["gpu_memory_utilization"] == 0.3, "Should use lower GPU memory for small models"


class TestServerConfigForPoolingModels:
    """Tests for server configuration with pooling models."""

    def test_embedding_model_vllm_args(self):
        """Test vLLM args for embedding models."""
        from cmw_vllm.server_config import ServerConfig

        config = ServerConfig(
            model="Qwen/Qwen3-Embedding-0.6B",
            port=8100,
            task="embed",
            runner="pooling",
        )

        args = config.to_vllm_args()
        args_str = " ".join(args)

        assert "--task embed" in args_str, "Should include --task embed"
        assert "--runner pooling" in args_str, "Should include --runner pooling"

    def test_reranker_model_vllm_args(self):
        """Test vLLM args for reranker models."""
        from cmw_vllm.server_config import ServerConfig

        config = ServerConfig(
            model="Qwen/Qwen3-Reranker-0.6B",
            port=8101,
            task="score",
            runner="pooling",
        )

        args = config.to_vllm_args()
        args_str = " ".join(args)

        assert "--task score" in args_str, "Should include --task score"
        assert "--runner pooling" in args_str, "Should include --runner pooling"

    def test_bge_reranker_with_hf_overrides(self):
        """Test vLLM args for BGE reranker with hf_overrides."""
        from cmw_vllm.server_config import ServerConfig

        config = ServerConfig(
            model="BAAI/bge-reranker-v2-m3",
            port=8102,
            task="score",
            runner="pooling",
            hf_overrides='{"architectures": ["BgeM3EmbeddingModel"]}',
        )

        args = config.to_vllm_args()
        args_str = " ".join(args)

        assert "--task score" in args_str, "Should include --task score"
        assert "--runner pooling" in args_str, "Should include --runner pooling"
        assert "--hf-overrides" in args_str, "Should include --hf-overrides"

    def test_pooling_models_from_registry_defaults(self):
        """Test that pooling models get correct defaults from registry."""
        from cmw_vllm.server_config import ServerConfig
        from cmw_vllm.model_registry import get_model_info

        test_models = [
            ("Qwen/Qwen3-Embedding-0.6B", "embed", "pooling"),
            ("Qwen/Qwen3-Reranker-0.6B", "score", "pooling"),
            ("BAAI/bge-reranker-v2-m3", "score", "pooling"),
            ("DiTy/cross-encoder-russian-msmarco", "score", "pooling"),
            ("ai-forever/FRIDA", "embed", "pooling"),
        ]

        for model_id, expected_task, expected_runner in test_models:
            model_info = get_model_info(model_id)
            assert model_info is not None, f"Model {model_id} not found in registry"

            # Simulate what CLI does when starting a model
            config = ServerConfig.from_env()
            config.model = model_id

            if "task" in model_info:
                config.task = model_info["task"]
            if "runner" in model_info:
                config.runner = model_info["runner"]
            if "dtype" in model_info:
                config.dtype = model_info["dtype"]
            if "max_model_len" in model_info:
                config.max_model_len = model_info["max_model_len"]
            if "gpu_memory_utilization" in model_info:
                config.gpu_memory_utilization = model_info["gpu_memory_utilization"]

            assert config.task == expected_task, f"Model {model_id}: task should be {expected_task}, got {config.task}"
            assert config.runner == expected_runner, f"Model {model_id}: runner should be {expected_runner}, got {config.runner}"

            args = config.to_vllm_args()
            args_str = " ".join(args)

            assert f"--task {expected_task}" in args_str, f"Model {model_id}: should include --task {expected_task}"
            assert f"--runner {expected_runner}" in args_str, f"Model {model_id}: should include --runner {expected_runner}"


class TestModelRegistryIntegrity:
    """Tests for model registry integrity."""

    def test_all_pooling_models_have_required_fields(self):
        """Test that all pooling models have required fields."""
        pooling_models = [
            "Qwen/Qwen3-Embedding-0.6B",
            "ai-forever/FRIDA",
            "Qwen/Qwen3-Reranker-0.6B",
            "BAAI/bge-reranker-v2-m3",
            "DiTy/cross-encoder-russian-msmarco",
        ]

        for model_id in pooling_models:
            model_info = get_model_info(model_id)
            assert model_info is not None, f"Model {model_id} not found in registry"
            assert "task" in model_info, f"Model {model_id}: missing 'task' field"
            assert "runner" in model_info, f"Model {model_id}: missing 'runner' field"
            assert "dtype" in model_info, f"Model {model_id}: missing 'dtype' field"
            assert model_info["runner"] == "pooling", f"Model {model_id}: runner should be 'pooling'"
            assert model_info["task"] in ["embed", "score"], f"Model {model_id}: task should be 'embed' or 'score'"

    def test_model_ids_are_unique(self):
        """Test that all model IDs in registry are unique."""
        model_ids = list(MODEL_REGISTRY.keys())
        assert len(model_ids) == len(set(model_ids)), "Model IDs should be unique"

    def test_model_info_has_required_fields(self):
        """Test that all models have required fields."""
        for model_id, model_info in MODEL_REGISTRY.items():
            assert "name" in model_info, f"Model {model_id}: missing 'name' field"
            assert "size_gb" in model_info, f"Model {model_id}: missing 'size_gb' field"
            assert "max_model_len" in model_info, f"Model {model_id}: missing 'max_model_len' field"
            assert "gpu_memory_utilization" in model_info, f"Model {model_id}: missing 'gpu_memory_utilization' field"
