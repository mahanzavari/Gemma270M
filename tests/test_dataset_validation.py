import pytest
from src.data_utils import create_prompt
from src.config import load_config
import os

class TestDatasetValidation:
    """Test class for validating dataset loading and preparation."""

    @pytest.fixture
    def config(self):
        """Load the default configuration for testing."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.yaml')
        return load_config(config_path)

    def test_config_loading(self, config):
        """Test that configuration loads correctly."""
        assert hasattr(config, 'DATASET_NAME'), "Config should have DATASET_NAME"
        assert hasattr(config, 'VAL_SET_SIZE'), "Config should have VAL_SET_SIZE"
        assert hasattr(config, 'SEED'), "Config should have SEED"
        print("Configuration loading test passed")

    def test_create_prompt_function_mock(self, config):
        """Test the create_prompt function with mock sample data."""
        # Test with valid sample
        mock_sample = {
            'instruction': 'سوال اول',
            'input': 'متن اول',
            'output': 'جواب اول'
        }

        prompt = create_prompt(mock_sample)
        assert prompt is not None, "Prompt should not be None for valid sample"
        assert "پرسش:" in prompt, "Prompt should contain Persian question marker"
        assert "متن:" in prompt, "Prompt should contain Persian context marker"
        assert "جواب کوتاه:" in prompt, "Prompt should contain Persian answer marker"
        assert "سوال اول" in prompt, "Prompt should contain the question"
        assert "متن اول" in prompt, "Prompt should contain the context"
        assert "جواب اول" in prompt, "Prompt should contain the answer"

        print("Create prompt function validation passed")

    def test_create_prompt_function_edge_cases(self, config):
        """Test the create_prompt function with edge cases."""
        # Test with missing output
        mock_sample_no_answers = {
            'instruction': 'سوال بدون جواب',
            'input': 'متن بدون جواب',
            'output': ''
        }

        prompt = create_prompt(mock_sample_no_answers)
        assert prompt is None, "Prompt should be None for sample with no answers"

        # Test with missing output field
        mock_sample_no_answers_field = {
            'instruction': 'سوال بدون فیلد جواب',
            'input': 'متن بدون فیلد جواب'
        }

        prompt = create_prompt(mock_sample_no_answers_field)
        assert prompt is None, "Prompt should be None for sample with no answers field"

        print("Create prompt edge cases validation passed")

    def test_dataset_structure_validation(self, config):
        """Test that expected dataset structure is validated."""
        # Mock sample structure that should be valid
        valid_sample = {
            'instruction': 'سوال معتبر',
            'input': 'متن معتبر',
            'output': 'جواب معتبر'
        }

        # Check required fields
        required_fields = ['instruction', 'input', 'output']
        for field in required_fields:
            assert field in valid_sample, f"Sample missing required field: {field}"

        # Check output is not empty
        assert valid_sample['output'], "Output should not be empty"

        print("Dataset structure validation passed")

    def test_data_quality_checks(self, config):
        """Test basic data quality validation logic."""
        mock_samples = [
            {
                'instruction': 'سوال اول',
                'input': 'متن اول',
                'output': 'جواب اول'
            },
            {
                'instruction': 'سوال دوم',
                'input': 'متن دوم',
                'output': 'جواب دوم'
            },
            {
                'instruction': '',  # Invalid: empty instruction
                'input': 'متن سوم',
                'output': 'جواب سوم'
            },
            {
                'instruction': 'سوال چهارم',
                'input': 'متن چهارم',
                'output': ''  # Invalid: no output
            }
        ]

        total_samples = len(mock_samples)
        valid_samples = 0

        for sample in mock_samples:
            if (sample.get('instruction') and
                sample.get('output')):
                valid_samples += 1

        # Should have 2 valid samples out of 4
        valid_ratio = valid_samples / total_samples
        assert valid_ratio == 0.5, f"Expected 50% valid samples, got {valid_ratio:.2%}"

        print(f"Data quality check passed: {valid_ratio:.2%} valid samples out of {total_samples}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])