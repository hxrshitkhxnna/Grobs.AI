
import asyncio
import logging
from app.services.llm_service import LLMService
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_fallback():
    # Force some providers to be unavailable by clearing their keys in memory
    # (Note: settings might be cached, so we modify the service's clients directly)
    
    service = LLMService(provider="openai")
    
    # Mocking failure for OpenAI
    service.openai_client = None
    service.async_openai_client = None
    
    # Mocking failure for Anthropic
    service.anthropic_client = None
    service.async_anthropic_client = None
    
    # Mocking failure for Google
    service.google_client = None
    
    print("\n--- Testing Text Generation Fallback ---")
    prompt = "Hello, this is a test."
    # Should fall back to local or final fallback
    result = await service.generate_text_async(prompt, use_cache=False)
    print(f"Provider used: {result.provider}")
    print(f"Content: {result.content}")
    print(f"Is Fallback: {result.is_fallback}")

    print("\n--- Testing Resume Parsing Heuristic Fallback ---")
    resume_prompt = "RESUME TEXT:\nJohn Doe\nSoftware Engineer\nEmail: john@example.com\nSkills: Python, Java"
    # Should fall back to heuristic resume parser
    result = await service.generate_structured_output_async(resume_prompt, schema={}, use_cache=False)
    print(f"Result Type: {type(result)}")
    print(f"Name extracted: {result.get('full_name')}")
    print(f"Email extracted: {result.get('email')}")

if __name__ == "__main__":
    asyncio.run(test_fallback())
