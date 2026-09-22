# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("qwen3")
class Qwen3ReasoningParser(ReasoningParser):
    """
    Reasoning parser for the Qwen3 model.

    The Qwen3 model uses <think>...</think> tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by <think> and </think> tokens from the model's
    output.

    IMPORTANT: When enable_thinking=True, the Qwen3 chat template appends
    ``<think>\n`` to the end of the prompt. This means the model's generated
    output does NOT start with ``<think>``; it starts directly with reasoning
    text followed by ``</think>`` and then the actual content. The parser
    must handle this case: when ``</think>`` is present but ``<think>`` is
    absent, everything before ``</think>`` is reasoning content.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        if (self.think_start_token_id is None
                or self.think_end_token_id is None):
            raise RuntimeError(
                "Qwen3 reasoning parser could not locate think start/end "
                "tokens in the tokenizer!")

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.think_end_token_id in input_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        if self.think_end_token_id not in input_ids[:-1]:
            return []
        else:
            return input_ids[input_ids.index(self.think_end_token_id) + 1:]

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content

        When Qwen3 chat template has enable_thinking=True, ``<think>`` is
        already part of the prompt so the generated output starts directly
        with reasoning tokens.  We detect this case (no ``<think>`` in
        previous or delta, no ``</think>`` yet) and route those tokens to
        reasoning_content.
        """
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (delta_token_ids[0] in [
                self.think_start_token_id, self.think_end_token_id
        ]):
            return None

        if self.think_start_token_id in previous_token_ids:
            if self.think_end_token_id in delta_token_ids:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                return DeltaMessage(reasoning_content=reasoning_content,
                                    content=content if content else None)
            elif self.think_end_token_id in previous_token_ids:
                # <think> in previous, </think> in previous,
                # reasoning content continues
                return DeltaMessage(content=delta_text)
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        elif self.think_start_token_id in delta_token_ids:
            if self.think_end_token_id in delta_token_ids:
                # <think> in delta, </think> in delta, extract reasoning
                start_index = delta_text.find(self.think_start_token)
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[start_index +
                                               len(self.think_start_token
                                                   ):end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                return DeltaMessage(reasoning_content=reasoning_content,
                                    content=content if content else None)
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        else:
            # -----------------------------------------------------------
            # Neither previous nor delta contains <think>.
            # This happens when enable_thinking=True and the chat template
            # already placed <think> at the end of the prompt.  The model
            # output starts directly with reasoning text.
            # -----------------------------------------------------------
            if self.think_end_token_id in previous_token_ids:
                # </think> already seen -> past reasoning, emit content.
                return DeltaMessage(content=delta_text)
            elif self.think_end_token_id in delta_token_ids:
                # </think> in this delta -> split reasoning / content.
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                return DeltaMessage(
                    reasoning_content=reasoning_content or None,
                    content=content if content else None)
            else:
                # No </think> seen yet -> still in reasoning phase.
                return DeltaMessage(reasoning_content=delta_text)

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from the model output.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content

        When ``enable_thinking=True``, the chat template places ``<think>``
        inside the prompt so the model output looks like ``abc</think>xyz``
        (no leading ``<think>``).  We handle that by treating everything
        before ``</think>`` as reasoning content.

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # --- Case 1: output contains both <think> and </think> ---
        if (self.think_start_token in model_output
                and self.think_end_token in model_output):
            # Strip <think> prefix
            model_output_parts = model_output.partition(
                self.think_start_token)
            model_output = (model_output_parts[2]
                            if model_output_parts[1]
                            else model_output_parts[0])
            # Split on </think>
            reasoning_content, _, content = model_output.partition(
                self.think_end_token)
            return reasoning_content, content or None

        # --- Case 2: output contains </think> but NOT <think> ---
        # Normal path when enable_thinking=True: the chat template already
        # emitted <think> as part of the prompt, so the model output starts
        # with reasoning text directly.
        if self.think_end_token in model_output:
            reasoning_content, _, content = model_output.partition(
                self.think_end_token)
            # Strip the leading newline that the chat template adds after
            # <think>\n - the model continues from there.
            if reasoning_content.startswith("\n"):
                reasoning_content = reasoning_content[1:]
            return reasoning_content or None, content.strip() or None

        # --- Case 2b: output contains <think> but NOT </think> ---
        # This happens when the model emits its own <think> tag (e.g. when
        # enable_thinking did not inject it into the prompt, or the model
        # echoes it anyway) and then gets truncated by max_tokens before
        # producing </think>.  Everything after <think> is reasoning;
        # there is no final content yet.
        if self.think_start_token in model_output:
            _, _, after_think = model_output.partition(
                self.think_start_token)
            reasoning_content = after_think.strip()
            return reasoning_content or None, None

        # --- Case 3: neither tag present ---
        # When enable_thinking=True, the chat template injects <think>\n at
        # the end of the prompt.  The model's output then starts directly
        # with reasoning text.  If we reach here it means the model was
        # truncated (finish_reason=length) before it could emit </think>.
        # In that case the entire output is reasoning, not content.
        _thinking_on = (
            request is not None
            and getattr(request, 'chat_template_kwargs', None) is not None
            and request.chat_template_kwargs.get('enable_thinking', False))
        if _thinking_on:
            reasoning_content = model_output.strip()
            return reasoning_content or None, None

        # thinking disabled or no reasoning block at all
        return None, model_output
