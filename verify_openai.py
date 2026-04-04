from __future__ import annotations
import asyncio
from core.config import config
from openai import AsyncOpenAI

async def test_api():
    if not config.validate():
        print("API key missing!")
        return
    client = AsyncOpenAI(api_key=config.API_KEY, base_url=config.API_BASE_URL)
    try:
        response = await client.models.list()
        print("Success! NVIDIA API (Kimi k2.5) is reachable.")
        print(f"Found {len(response.data)} models.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_api())