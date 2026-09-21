import base64
from typing import Optional
from src.domain.models import PhysicalImageReference, SemanticDescription
from src.domain.ports import VisionEncoderPort

# Try to import globally for patching, fallback if not available
try:
    from google import genai
except ImportError:
    genai = None

class GeminiExternalAdapter(VisionEncoderPort):
    """
    Delegates the VLM tensor mapping to the Google Gemini API using `google-genai`.
    This maps continuous bytes into an discrete string payload.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def encode_manifold(self, image: PhysicalImageReference) -> SemanticDescription:
        # Prevent allocation if file doesn't exist
        if not image.file_path.exists():
            raise FileNotFoundError(f"Missing manifold: {image.file_path}")

        image_bytes = image.file_path.read_bytes()

        # Enforce Lazy Loading import boundary
        if genai is None:
            raise ImportError("The 'google-genai' library is required to execute the Gemini external adapter.")

        client = genai.Client(api_key=self.api_key)

        # Translate the binary spatial tensor to standard base64 for network transit
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = "image/png"
        if image.file_path.suffix.lower() in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"

        # Semantic mapping prompt
        prompt = (
            "Analyze this image. If it is a diagram or figure, provide a concise ALT text description. "
            "If it contains text or math that is part of a document, transcribe it perfectly as Markdown. "
            "Return ONLY the requested text/markdown, with no conversational filler."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                prompt,
                {
                    "mime_type": mime_type,
                    "data": b64_data
                }
            ]
        )

        semantic_str = response.text if response.text else "[ALT Text] Encoding Failed"

        return SemanticDescription(
            content=semantic_str,
            metadata={"adapter": "GeminiExternalAdapter", "model": "gemini-2.5-flash"}
        )
