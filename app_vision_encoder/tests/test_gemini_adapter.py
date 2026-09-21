import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.adapters.gemini_adapter import GeminiExternalAdapter
from src.domain.models import PhysicalImageReference

@patch("src.adapters.gemini_adapter.genai")
def test_gemini_adapter_encodes_manifold_successfully(mock_genai, tmp_path: Path):
    # Setup the physical test tensor
    file_path = tmp_path / "diagram.png"
    file_path.write_bytes(b"dummy_png_bytes")
    image_ref = PhysicalImageReference(file_path=file_path, file_size_bytes=15)

    # Mock the API client and response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "A detailed diagram of a neural network."
    mock_client.models.generate_content.return_value = mock_response
    mock_genai.Client.return_value = mock_client

    # Execute mapping
    adapter = GeminiExternalAdapter(api_key="fake-key")
    result = adapter.encode_manifold(image_ref)

    assert result.content == "A detailed diagram of a neural network."
    assert result.metadata["adapter"] == "GeminiExternalAdapter"

    # Verify exact network transit payload was formulated correctly
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["contents"][1]["mime_type"] == "image/png"

def test_gemini_adapter_raises_on_missing_file():
    fake_path = Path("/tmp/does_not_exist.png")
    # domain model validates existence upon init, so we must mock the model or instantiate carefully
    with pytest.raises(FileNotFoundError):
        PhysicalImageReference(file_path=fake_path, file_size_bytes=0)
