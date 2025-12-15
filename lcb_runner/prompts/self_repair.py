import json

from anthropic import HUMAN_PROMPT, AI_PROMPT

from lcb_runner.lm_styles import LMStyle


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once."

    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you are helping a user correct a error program for code competition. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the entire executable program. You must put the entire fixed executable program within code delimiters."

    SYSTEM_MESSAGE_CODEQWEN = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    )

    SYSTEM_MESSAGE_QWEN_QWQ = f"<|im_start|>system\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\n<|im_start|>user"

    FORMATTING_REPEAT = f"First reason about the code providing a textual explanation of what is wrong with the code and then generate a fixed of the program enclosed code delimiters."

    FORMATTING_MESSAGE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."


# def truncate_io(io):
#     if len(str(io)) > 200:
#         io = str(io)[:200] + "...."
#     return io


def get_check_prompt(question: str, result, metadata):
    ## assumes i/o examples are already truncated!
    ## less pressure on storing 10 MB json because on a single large input-output pair
    # result_by_test_case = result
    # assert len(metadata) == 1, f"metadata = {metadata}"
    # metadata = metadata[0]
    metadata = json.loads(metadata)
    if "error_code" not in metadata:
        return ""
    if metadata["error_code"] == -1:
        # compilation error
        message = f"The above code is incorrect and got the following compilation error.\n{metadata.get('error', 'Unknown error')}"
    elif metadata["error_code"] == -2:
        # wrong answer
        inputs_str = f"Input: {metadata['inputs']}\n" if 'inputs' in metadata else ""
        output_str = f"Generated Output: {metadata['output']}\n" if 'output' in metadata else ""
        expected_str = f"Expected: {metadata['expected']}" if 'expected' in metadata else ""
        message = f"The above code is incorrect and got a wrong answer.\n{inputs_str}{output_str}{expected_str}"
    elif metadata["error_code"] == -3:
        # time limit exceeded
        error_str = metadata.get('error', '')
        inputs_str = f"\nInput: {metadata['inputs']}" if 'inputs' in metadata else ""
        expected_str = f"\nExpected: {metadata['expected']}" if 'expected' in metadata else ""
        message = f"The above code is incorrect and got time limit exceeded.\n{error_str}{inputs_str}{expected_str}"
    elif metadata["error_code"] == -4:
        # runtime error
        error_str = metadata.get('error', metadata.get('error_message', 'Unknown runtime error'))
        inputs_str = f"\nInput: {metadata['inputs']}" if 'inputs' in metadata else ""
        expected_str = f"\nExpected: {metadata['expected']}" if 'expected' in metadata else ""
        message = f"The above code is incorrect and got a runtime error.{inputs_str}{expected_str}\n{error_str}"
    else:
        # handle any other error codes (e.g., -5, -6, etc.)
        error_str = metadata.get('error', metadata.get('error_message', f"Error code {metadata.get('error_code', 'unknown')}"))
        inputs_str = f"\nInput: {metadata['inputs']}" if 'inputs' in metadata else ""
        expected_str = f"\nExpected: {metadata['expected']}" if 'expected' in metadata else ""
        output_str = f"\nGenerated Output: {metadata['output']}" if 'output' in metadata else ""
        message = f"The above code is incorrect and encountered an error.{inputs_str}{output_str}{expected_str}\n{error_str}"
    return message


def get_generic_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Question:\n{question}\n\n"
    prompt += f"### Answer:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata) + "\n"
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_cllama_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Question\n{question}\n\n"
    prompt += f"### Answer\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_deepseekcode_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Instruction: You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question}\n\n"
    prompt += f"### Response:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_codeqwen_question_template_answer(question: str, code, result, metadata):
    prompt = "You will be given a question (problem specification) and a Python program that has errors. You will analyze the errors and generate a correct Python program that fixes the issues.\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += f"Incorrect Code:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"\n\n### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def get_qwen_qwq_question_template_answer(question: str, code, result, metadata):
    prompt = "You will be given a question (problem specification) and a Python program that has errors. Analyze the errors step-by-step and generate a correct Python program that fixes the issues.\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += f"Incorrect Code:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"\n\n### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def format_prompt_self_repair(
    question: str, LanguageModelStyle: LMStyle, code, result, metadata
) -> str:
    if result:
        # The code is accepted, no need to change anything.
        return ""
    if LanguageModelStyle == LMStyle.OpenAIChat:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                )
                + "\n\n"
                + PromptConstants.FORMATTING_REPEAT,
            },
        ]
        return chat_messages
    if LanguageModelStyle == LMStyle.LLaMa3:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ),
            },
        ]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
    elif LanguageModelStyle == LMStyle.Claude:
        prompt = f"{HUMAN_PROMPT}\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n{get_generic_question_template_answer(question, code, result, metadata).rstrip()}\n{AI_PROMPT}"
        return prompt
    elif LanguageModelStyle == LMStyle.Claude3:
        system = PromptConstants.SYSTEM_MESSAGE_GENERIC
        prompt = [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ).rstrip(),
            }
        ]
        return system, prompt
    elif LanguageModelStyle == LMStyle.MistralWeb:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question, code, result, metadata),
            },
        ]
        return chat_messages
    elif LanguageModelStyle == LMStyle.Gemini:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{get_generic_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.StarCoderInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{get_generic_question_template_answer(question, code, result,metadata)}"
        return prompt

    elif LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK}\n\n{get_deepseekcode_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.CodeLLaMaInstruct:
        prompt = f"[INST] <<SYS>>\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n<</SYS>>\n\n{get_cllama_question_template_answer(question, code, result,metadata)}\n[/INST]"
        return prompt
    elif LanguageModelStyle == LMStyle.CodeQwenInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CODEQWEN}\n\n"
        prompt += f"{get_codeqwen_question_template_answer(question, code, result, metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.QwQ:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_QWEN_QWQ}\n\n"
        prompt += f"{get_qwen_qwq_question_template_answer(question, code, result, metadata)}"
        return prompt
    else:
        raise NotImplementedError(
            f"LanguageModelStyle {LanguageModelStyle} not implemented"
        )


def extract_code(model_output: str, lmstyle: LMStyle):
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def test():
    def write_str_or_json(prompt):
        if isinstance(prompt, str):
            fp.write(prompt)
        else:
            fp.write(json.dumps(prompt))
        return

    for lm_style in [LMStyle.OpenAIChat]:
        with open(
            "output/GPT-3.5-Turbo-0125/Scenario.codegeneration_10_0.2_eval_all.json"
        ) as f:
            check_metadata = json.load(f)[0]
        checked_base_question_cotent = check_metadata["question_content"]
        checked_base_codes = check_metadata["code_list"][0]
        checked_base_results = check_metadata["graded_list"][0]
        checked_base_metadata = check_metadata["metadata"][0]
        leetcode_prompt = format_prompt_self_repair(
            checked_base_question_cotent,
            lm_style,
            checked_base_codes,
            checked_base_results,
            checked_base_metadata,
        )

        with open(f"/tmp/leetcode_{lm_style}.txt", "w") as fp:
            write_str_or_json(leetcode_prompt)
    return


if __name__ == "__main__":
    test()
