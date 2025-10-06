import pytest
from unittest.mock import Mock, MagicMock, patch
import time
from src.callbacks import TrainingLoggerCallback, EarlyStoppingCallback
from src.metrics import compute_metrics, log_training_metrics, get_model_memory_usage, log_system_info
from transformers import TrainingArguments, TrainerState, TrainerControl


class TestCallbacks:
    """Test class for custom training callbacks."""

    def test_training_logger_callback_initialization(self):
        """Test TrainingLoggerCallback initialization."""
        callback = TrainingLoggerCallback(log_every_n_steps=10)
        assert callback.log_every_n_steps == 10
        assert callback.start_time is None
        assert callback.last_log_time is None

    def test_training_logger_callback_on_train_begin(self):
        """Test on_train_begin method."""
        callback = TrainingLoggerCallback()
        args = Mock(spec=TrainingArguments)
        args.output_dir = "test_output"
        args.per_device_train_batch_size = 4
        args.gradient_accumulation_steps = 2
        args.max_steps = 100

        state = Mock(spec=TrainerState)
        state.max_steps = 100

        control = Mock(spec=TrainerControl)

        # Mock logging to capture output
        import logging
        with patch('src.callbacks.logging') as mock_logging:
            callback.on_train_begin(args, state, control)

            # Check that logging.info was called
            assert mock_logging.info.called
            # Check that start_time and last_log_time are set
            assert callback.start_time is not None
            assert callback.last_log_time is not None

    def test_training_logger_callback_on_step_end(self):
        """Test on_step_end method."""
        callback = TrainingLoggerCallback(log_every_n_steps=5)
        callback.start_time = time.time() - 10  # 10 seconds ago
        callback.last_log_time = callback.start_time

        args = Mock(spec=TrainingArguments)
        state = Mock(spec=TrainerState)
        state.global_step = 5
        state.max_steps = 100
        state.log_history = [{'train_loss': 0.5, 'learning_rate': 0.001}]

        control = Mock(spec=TrainerControl)

        with patch('src.callbacks.logging') as mock_logging:
            callback.on_step_end(args, state, control)

            # Should log at step 5 (multiple of 5)
            assert mock_logging.info.called

    def test_early_stopping_callback_initialization(self):
        """Test EarlyStoppingCallback initialization."""
        callback = EarlyStoppingCallback(patience=5, min_delta=0.01)
        assert callback.patience == 5
        assert callback.min_delta == 0.01
        assert callback.best_loss == float('inf')
        assert callback.counter == 0

    def test_early_stopping_callback_on_evaluate_improvement(self):
        """Test early stopping with loss improvement."""
        callback = EarlyStoppingCallback(patience=3, min_delta=0.01)

        args = Mock(spec=TrainingArguments)
        state = Mock(spec=TrainerState)
        control = Mock(spec=TrainerControl)

        # First evaluation - set best loss
        metrics1 = {'eval_loss': 0.5}
        callback.on_evaluate(args, state, control, metrics1)
        assert callback.best_loss == 0.5
        assert callback.counter == 0
        assert control.should_training_stop is not True

        # Second evaluation - improvement
        metrics2 = {'eval_loss': 0.4}
        callback.on_evaluate(args, state, control, metrics2)
        assert callback.best_loss == 0.4
        assert callback.counter == 0

    def test_early_stopping_callback_trigger(self):
        """Test early stopping trigger."""
        callback = EarlyStoppingCallback(patience=2, min_delta=0.01)

        args = Mock(spec=TrainingArguments)
        state = Mock(spec=TrainerState)
        control = Mock(spec=TrainerControl)

        # Set initial best loss
        metrics1 = {'eval_loss': 0.5}
        callback.on_evaluate(args, state, control, metrics1)

        # No improvement - counter increases
        metrics2 = {'eval_loss': 0.51}
        callback.on_evaluate(args, state, control, metrics2)
        assert callback.counter == 1

        # Still no improvement - counter increases again
        metrics3 = {'eval_loss': 0.52}
        callback.on_evaluate(args, state, control, metrics3)
        assert callback.counter == 2
        assert control.should_training_stop


class TestMetrics:
    """Test class for metrics computation functions."""

    def test_compute_metrics(self):
        """Test compute_metrics function."""
        import torch

        # Mock evaluation predictions and labels as a tuple
        # For causal LM: predictions should be (batch_size, seq_len, vocab_size)
        # labels should be (batch_size, seq_len)
        batch_size, seq_len, vocab_size = 2, 5, 100
        predictions = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        eval_pred = (predictions, labels)

        # Mock the torch operations that might be problematic
        with patch('src.metrics.torch.nn.CrossEntropyLoss') as mock_loss_cls, \
             patch('src.metrics.torch.exp') as mock_exp, \
             patch('src.metrics.torch.argmax') as mock_argmax:

            # Mock CrossEntropyLoss instance
            mock_loss_instance = Mock()
            mock_loss_instance.return_value = torch.tensor(0.5)
            mock_loss_cls.return_value = mock_loss_instance

            # Mock exp
            mock_exp.return_value = torch.tensor(1.648721)

            # Mock argmax
            mock_argmax.return_value = torch.randint(0, vocab_size, (batch_size, seq_len-1))

            metrics = compute_metrics(eval_pred)

            assert 'eval_loss' in metrics
            assert 'eval_perplexity' in metrics
            assert isinstance(metrics['eval_loss'], float)
            assert isinstance(metrics['eval_perplexity'], float)

    def test_log_training_metrics(self):
        """Test log_training_metrics function."""
        metrics = {'loss': 0.5, 'accuracy': 0.85}
        trainer = Mock()

        with patch('src.metrics.logging') as mock_logging:
            log_training_metrics(trainer, metrics)

            # Check that logging.info was called
            assert mock_logging.info.called

    @patch('src.metrics.torch.cuda.is_available')
    @patch('src.metrics.torch.cuda.memory_allocated')
    @patch('src.metrics.torch.cuda.memory_reserved')
    def test_get_model_memory_usage_with_cuda(self, mock_reserved, mock_allocated, mock_cuda_available):
        """Test get_model_memory_usage when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_allocated.return_value = 1024**3  # 1 GB
        mock_reserved.return_value = 2 * 1024**3  # 2 GB

        result = get_model_memory_usage()

        assert 'gpu_memory_allocated_gb' in result
        assert 'gpu_memory_reserved_gb' in result
        assert result['gpu_memory_allocated_gb'] == 1.0
        assert result['gpu_memory_reserved_gb'] == 2.0

    @patch('src.metrics.torch.cuda.is_available')
    def test_get_model_memory_usage_without_cuda(self, mock_cuda_available):
        """Test get_model_memory_usage when CUDA is not available."""
        mock_cuda_available.return_value = False

        result = get_model_memory_usage()

        assert result == {}

    def test_log_system_info(self):
        """Test log_system_info function."""
        with patch('src.metrics.logging') as mock_logging, \
             patch('src.metrics.torch') as mock_torch:

            # Mock system info
            mock_torch.__version__ = "2.0.0"
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.cuda.get_device_name.return_value = "Tesla V100"
            mock_torch.get_num_threads.return_value = 8

            log_system_info()

            # Check that logging.info was called multiple times
            assert mock_logging.info.call_count >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])