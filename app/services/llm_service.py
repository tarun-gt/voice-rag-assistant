
import httpx


class LLMService:

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:latest",
        timeout: float = 180.0
    ):

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_response(self, prompt: str) -> str:

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "think": False,
            "stream": False
        }

        try:

            response = httpx.post(
                url,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()

            answer = result.get("message", {}).get("content", "")

            if not answer or not answer.strip():
                raise RuntimeError(
                    "Ollama returned an empty response"
                )

            return answer.strip()

        except httpx.ConnectError as error:

            raise RuntimeError(
                "Unable to connect to Ollama. "
                "Make sure Ollama is running."
            ) from error

        except httpx.TimeoutException as error:

            raise RuntimeError(
                "Ollama request timed out."
            ) from error

        except httpx.HTTPStatusError as error:

            raise RuntimeError(
                f"Ollama returned HTTP error: "
                f"{error.response.status_code}"
            ) from error
