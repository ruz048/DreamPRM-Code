import os
import re
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner


class OpenAIRunner(BaseRunner):
    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
    )

    # Patterns that indicate the model is refusing to provide a solution
    REFUSAL_PATTERNS = [
        r"I'm sorry,?\s+but\s+I\s+can'?t\s+provide\s+a\s+solution",
        r"I'm sorry,?\s+but\s+I\s+cannot\s+provide\s+a\s+solution",
        r"I'?m\s+unable\s+to\s+(provide|solve|help\s+with)\s+(this|that|the)",
        r"I\s+can'?t\s+(assist|help)\s+with\s+(this|that)",
        r"I'm sorry,?\s+but\s+I\s+can'?t\s+(assist|help)",
        r"I\s+cannot\s+(assist|help|provide)\s+with",
        r"I'm sorry,?\s+I\s+can'?t\s+(provide|solve|help)",
    ]

    def __init__(self, args, model):
        super().__init__(args, model)
        self.num_completions = args.n  # Store n separately for manual looping
        self.max_refusal_retries = 10  # Maximum number of retries for refusal responses
        if model.model_style == LMStyle.OpenAIReasonPreview:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "max_completion_tokens": 25000,
            }
            self.use_n_parameter = False  # Reasoning models don't support n parameter
        elif model.model_style == LMStyle.OpenAIReason:
            assert (
                "__" in args.model
            ), f"Model {args.model} is not a valid OpenAI Reasoning model as we require reasoning effort in model name."
            model, reasoning_effort = args.model.split("__")
            self.client_kwargs: dict[str | str] = {
                "model": model,
                "reasoning_effort": reasoning_effort,
            }
            self.use_n_parameter = False  # Reasoning models don't support n parameter
        else:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": args.openai_timeout,
                # "stop": args.stop, --> stop is only used for base models currently
            }
            self.use_n_parameter = True  # Standard models support n parameter

    @staticmethod
    def is_refusal_response(response_text: str) -> bool:
        """
        Check if the response contains refusal patterns.
        Returns True if the model is refusing to provide a solution.
        """
        if not response_text or not isinstance(response_text, str):
            return False

        # Check against each refusal pattern
        for pattern in OpenAIRunner.REFUSAL_PATTERNS:
            if re.search(pattern, response_text, re.IGNORECASE):
                return True
        return False

    def _run_single(self, prompt: list[dict[str, str]], n: int = 10, refusal_retry_count: int = 0) -> list[str]:
        assert isinstance(prompt, list)

        if n == 0:
            print("Max retries reached. Returning empty response.")
            return []

        # If the model supports n parameter, use it directly
        if self.use_n_parameter:
            try:
                response = OpenAIRunner.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.OpenAIError,
                openai.APIStatusError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.APIConnectionError,
            ) as e:
                print("Exception: ", repr(e))
                print("Sleeping for 30 seconds...")
                print("Consider reducing the number of parallel processes.")
                sleep(30)
                return self._run_single(prompt, n=n - 1, refusal_retry_count=refusal_retry_count)
            except Exception as e:
                print(f"Failed to run the model for {prompt}!")
                print("Exception: ", repr(e))
                raise e

            # Extract responses
            responses = [c.message.content for c in response.choices]

            # Check for refusals and retry if necessary
            refusal_detected = any(self.is_refusal_response(resp) for resp in responses)
            if refusal_detected and refusal_retry_count < self.max_refusal_retries:
                print(f"Refusal response detected (attempt {refusal_retry_count + 1}/{self.max_refusal_retries}). Retrying...")
                return self._run_single(prompt, n=n, refusal_retry_count=refusal_retry_count + 1)
            elif refusal_detected:
                print(f"Max refusal retries ({self.max_refusal_retries}) reached. Returning responses anyway.")

            return responses

        # For models that don't support n parameter, loop manually
        else:
            all_responses = []
            for i in range(self.num_completions):
                completion_success = False
                current_refusal_retry = 0

                # Retry loop for this specific completion
                while not completion_success and current_refusal_retry <= self.max_refusal_retries:
                    try:
                        response = OpenAIRunner.client.chat.completions.create(
                            messages=prompt,
                            **self.client_kwargs,
                        )
                        response_content = response.choices[0].message.content

                        # Check for refusal
                        if self.is_refusal_response(response_content):
                            if current_refusal_retry < self.max_refusal_retries:
                                current_refusal_retry += 1
                                print(f"Refusal detected in completion {i+1}/{self.num_completions} "
                                      f"(attempt {current_refusal_retry}/{self.max_refusal_retries}). Retrying...")
                                continue
                            else:
                                print(f"Max refusal retries reached for completion {i+1}. Using response anyway.")
                                all_responses.append(response_content)
                                completion_success = True
                        else:
                            all_responses.append(response_content)
                            completion_success = True

                    except (
                        openai.APIError,
                        openai.RateLimitError,
                        openai.InternalServerError,
                        openai.OpenAIError,
                        openai.APIStatusError,
                        openai.APITimeoutError,
                        openai.InternalServerError,
                        openai.APIConnectionError,
                    ) as e:
                        print(f"Exception on completion {i+1}/{self.num_completions}: ", repr(e))
                        print("Sleeping for 30 seconds...")
                        print("Consider reducing the number of parallel processes.")
                        sleep(30)
                        # Retry this specific completion via main retry mechanism
                        return self._run_single(prompt, n=n - 1, refusal_retry_count=refusal_retry_count)
                    except Exception as e:
                        print(f"Failed to run the model for {prompt}!")
                        print("Exception: ", repr(e))
                        raise e

            return all_responses
