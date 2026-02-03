"""GLM-OCR Processor with OCR-specific templates.

GLM-OCR uses the same image processing and tokenization as GLM-4V,
but provides specialized prompt templates for document understanding tasks.
"""

from ..glm4v.processing import Glm46VProcessor


class GlmOcrProcessor(Glm46VProcessor):
    """Processor for GLM-OCR model.
    
    Inherits from GLM-4V processor with OCR-specific chat templates.
    GLM-OCR is optimized for:
    - Text extraction from documents
    - Layout understanding (tables, forms)
    - Formula and code recognition
    - Structured data extraction
    """
    
    # Default OCR chat template for structured extraction
    OCR_CHAT_TEMPLATE = """
    {%- for message in messages %}
        {%- if message['role'] == 'user' %}
            {{ '<|user|>\n' + message['content'] + '<|endoftext|>\n' }}
        {%- elif message['role'] == 'assistant' %}
            {{ '<|assistant|>\n' + message['content'] + '<|endoftext|>\n' }}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{ '<|assistant|>\n' }}
    {%- endif %}
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store default OCR system prompt
        self.default_ocr_prompt = "Extract all text from this image accurately."
    
    def get_ocr_prompt(self, task="extract_text", structure=None):
        """Generate OCR-specific prompts.
        
        Args:
            task: Type of OCR task (extract_text, table, form, formula)
            structure: Optional JSON schema for structured output
            
        Returns:
            Formatted prompt string
        """
        prompts = {
            "extract_text": "Extract all text content from this image accurately, preserving the layout and formatting as much as possible.",
            "table": "Extract the table from this image. Return the content in a structured format with rows and columns preserved.",
            "form": "Extract all fields and values from this form. Return as key-value pairs.",
            "formula": "Extract and transcribe any mathematical formulas or equations from this image using LaTeX notation.",
            "document": "Extract all content from this document including text, tables, and structure. Preserve the document hierarchy.",
        }
        
        base_prompt = prompts.get(task, prompts["extract_text"])
        
        if structure:
            base_prompt += f"\nReturn the result in the following JSON structure: {structure}"
        
        return base_prompt


__all__ = ["GlmOcrProcessor"]
