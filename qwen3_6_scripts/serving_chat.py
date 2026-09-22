# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Callable, Final, Optional, Union

import jinja2
import partial_json_parser
from fastapi import Request
from pydantic import TypeAdapter

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption,
                                         ConversationMessage)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb, ChatCompletionLogProbs,
    ChatCompletionLogProbsContent, ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaFunctionCall, DeltaMessage,
    DeltaToolCall, ErrorResponse, FunctionCall, FunctionDefinition,
    PromptTokenUsageInfo, RequestResponseMetadata, ToolCall, UsageInfo)
import re as _re
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    clamp_prompt_logprobs)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import (
    MistralToolCall)
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.transformers_utils.tokenizers import (maybe_serialize_tool_calls,
                                                truncate_tool_call_ids)
from vllm.utils import random_uuid

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# [9071dd22] Minja / Jinja2 compat: rewrite "is undefined" → "is none"
# ---------------------------------------------------------------------------
def _normalize_undefined_tests(template: str) -> str:
    """Rewrite 'is undefined' → 'is none' inside Jinja2 blocks."""
    def _replace_in_block(block_interior: str) -> str:
        result = []
        in_quote = None
        i = 0
        while i < len(block_interior):
            ch = block_interior[i]
            if not in_quote and ch in ("'", '"'):
                in_quote = ch
                result.append(ch)
                i += 1
                continue
            if in_quote:
                if ch == '\\' and i + 1 < len(block_interior):
                    result.append(ch)
                    result.append(block_interior[i + 1])
                    i += 2
                    continue
                if ch == in_quote:
                    in_quote = None
                result.append(ch)
                i += 1
                continue
            tail = block_interior[i:]
            if tail.startswith("is not undefined"):
                result.append("is not none")
                i += len("is not undefined")
            elif tail.startswith("is undefined"):
                result.append("is none")
                i += len("is undefined")
            else:
                result.append(ch)
                i += 1
        return "".join(result)

    out = []
    i = 0
    while i < len(template):
        if (i + 1 < len(template) and template[i] == '{'
                and template[i + 1] in ('{', '%')):
            is_expr = (template[i + 1] == '{')
            closer = "}}" if is_expr else "%}"
            end = template.find(closer, i + 2)
            if end == -1:
                out.append(template[i:])
                break
            out.append(template[i:i + 2])
            out.append(_replace_in_block(template[i + 2:end]))
            out.append(closer)
            i = end + 2
        else:
            out.append(template[i])
            i += 1
    return "".join(out)


def _serialize_tool_arguments(arguments) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, (dict, list)):
        return json.dumps(arguments, ensure_ascii=False)
    return json.dumps(arguments, ensure_ascii=False)


def _tool_arguments_are_json_object(arguments: str) -> bool:
    try:
        value = json.loads(arguments)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return isinstance(value, dict)


def _reclassify_named_guided_json(
    reasoning_text: Optional[str],
    output_text: str,
) -> tuple[Optional[str], str]:
    """Recover guided JSON misclassified as unterminated reasoning."""
    if (not output_text and reasoning_text is not None
            and _tool_arguments_are_json_object(reasoning_text)):
        return None, reasoning_text
    return reasoning_text, output_text


def _select_named_tool_arguments(
    output_text: str,
    expected_name: str,
    parsed_tool_calls: Optional[list[ToolCall]],
) -> str:
    """Use parser output only to repair a malformed named-tool payload."""
    if _tool_arguments_are_json_object(output_text):
        return output_text
    if not parsed_tool_calls or len(parsed_tool_calls) != 1:
        return output_text
    call = parsed_tool_calls[0]
    function = getattr(call, "function", None)
    if function is None or getattr(function, "name", None) != expected_name:
        return output_text
    arguments = _serialize_tool_arguments(
        getattr(function, "arguments", None))
    if not _tool_arguments_are_json_object(arguments):
        return output_text
    return arguments


def _named_tool_delta_payload(name: str, arguments: str, index: int,
                              call_id: str, first_delta: bool
                              ) -> dict[str, object]:
    function: dict[str, object] = {"arguments": arguments}
    payload: dict[str, object] = {"index": index, "function": function}
    if first_delta:
        function["name"] = name
        payload["id"] = call_id
        payload["type"] = "function"
    return payload


def _consume_named_tool_header_slot(header_sent: list[bool],
                                    index: int) -> bool:
    first_delta = not header_sent[index]
    header_sent[index] = True
    return first_delta


def _sequential_greedy_fanout_count(
    request: ChatCompletionRequest,
    max_num_seqs: int,
) -> int:
    """Return the supported deterministic fan-out width, or zero."""
    n = request.n if request.n is not None else 1
    if (
        n == 2
        and request.temperature == 0
        and not request.stream
        and not request.use_beam_search
        and request.best_of is None
        and request.prompt_logprobs is None
    ):
        return n
    return 0


def _merge_sequential_chat_responses(
    responses: list[ChatCompletionResponse],
    request_id: str,
    created_time: int,
) -> ChatCompletionResponse:
    if len(responses) != 2:
        raise ValueError("deterministic fan-out requires exactly two responses")
    first = responses[0]
    if any(response.model != first.model for response in responses):
        raise ValueError("fan-out response models differ")
    if any(len(response.choices) != 1 for response in responses):
        raise ValueError("fan-out child response must contain one choice")
    if any(response.usage.prompt_tokens != first.usage.prompt_tokens
           for response in responses):
        raise ValueError("fan-out prompt token counts differ")
    if any(response.usage.completion_tokens is None
           for response in responses):
        raise ValueError("fan-out completion token count is missing")
    choices = [
        response.choices[0].model_copy(
            deep=True, update={"index": index})
        for index, response in enumerate(responses)
    ]
    completion_tokens = sum(
        response.usage.completion_tokens or 0 for response in responses)
    reasoning_counts = [
        response.usage.reasoning_tokens for response in responses
    ]
    reasoning_tokens = (
        None if all(v is None for v in reasoning_counts)
        else sum(v or 0 for v in reasoning_counts))
    prompt_details = first.usage.prompt_tokens_details
    usage = UsageInfo(
        prompt_tokens=first.usage.prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=first.usage.prompt_tokens + completion_tokens,
        reasoning_tokens=reasoning_tokens,
        prompt_tokens_details=(
            prompt_details.model_copy(deep=True)
            if prompt_details is not None else None),
    )
    return ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=first.model,
        choices=choices,
        usage=usage,
        prompt_logprobs=first.prompt_logprobs,
    )


class OpenAIServingChat(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        enable_reasoning: bool = False,
        reasoning_parser: Optional[str] = None,
        enable_auto_tools: bool = False,
        tool_parser: Optional[str] = None,
        enable_prompt_tokens_details: bool = False,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        if self.enable_auto_tools:
            logger.info(
                "\"auto\" tool choice has been enabled please note that while"
                " the parallel_tool_calls client option is preset for "
                "compatibility reasons, it will be ignored.")

        self.enable_reasoning: bool = enable_reasoning
        self.reasoning_parser: Optional[Callable[[AnyTokenizer],
                                                 ReasoningParser]] = None
        if self.enable_reasoning:
            try:
                self.reasoning_parser = (
                    ReasoningParserManager.get_reasoning_parser(
                        reasoning_parser))
            except Exception as e:
                raise TypeError("Error: --enable-reasoning requires "
                                f"reasoning_parser:'{reasoning_parser}' "
                                "which has not been registered") from e
        self.tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None
        if self.enable_auto_tools:
            try:
                if (tool_parser == "pythonic" and
                        model_config.model.startswith("meta-llama/Llama-3.2")):
                    logger.warning(
                        "Llama3.2 models may struggle to emit valid pythonic"
                        " tool calls")
                self.tool_parser = ToolParserManager.get_tool_parser(
                    tool_parser)
            except Exception as e:
                raise TypeError("Error: --enable-auto-tool-choice requires "
                                f"tool_parser:'{tool_parser}' which has not "
                                "been registered") from e

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self._template_patched = False  # [BI100] one-shot chat template patch
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default chat sampling params from %s: %s",
                        source, self.default_sampling_params)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse,
               ErrorResponse]:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if not request.messages:
            return self.create_error_response(
                "messages must contain at least one message")

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        # [BI100] Sequential greedy fan-out for n=2 with max_num_seqs=1
        if request.n is not None and request.n > 1:
            scheduler_config = await self.engine_client.get_scheduler_config()
            max_num_seqs = scheduler_config.max_num_seqs
            if request.n > max_num_seqs:
                fanout_count = _sequential_greedy_fanout_count(
                    request, max_num_seqs)
                if fanout_count:
                    return await self._create_sequential_greedy_fanout(
                        request, raw_request, fanout_count)
                return self.create_error_response(
                    f"n={request.n} exceeds max_num_seqs={max_num_seqs}. "
                    f"Use n<={max_num_seqs} or omit n.")

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            model_name = self._get_model_name(request.model, lora_request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            tool_parser = self.tool_parser

            if isinstance(tokenizer, MistralTokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                maybe_serialize_tool_calls(request)
                truncate_tool_call_ids(request)

            # [BI100] Runtime chat template patch (one-shot).
            if not self._template_patched and hasattr(tokenizer, 'chat_template'):
                if isinstance(tokenizer.chat_template, str):
                    _tgt = r"{{- '<think>\n\n</think>\n\n' }}"
                    if _tgt in tokenizer.chat_template:
                        tokenizer.chat_template = tokenizer.chat_template.replace(
                            _tgt, "{{- '' }}", 1)
                        logger.info("[BI100] Patched chat_template: removed empty "
                                    "<think></think> block for non-thinking mode")
                    if "is undefined" in tokenizer.chat_template:
                        tokenizer.chat_template = _normalize_undefined_tests(
                            tokenizer.chat_template)
                        logger.info("[BI100] Patched chat_template: rewrote "
                                    "'is undefined' → 'is none' for Minja compat")
                self._template_patched = True

            # [BI100] Ensure enable_thinking defaults to True
            if request.chat_template_kwargs is None:
                request.chat_template_kwargs = {}
            if "enable_thinking" not in request.chat_template_kwargs:
                request.chat_template_kwargs["enable_thinking"] = True

            if (request.tool_choice == "auto" and
                    not (self.enable_auto_tools and tool_parser is not None)
                    and not isinstance(tokenizer, MistralTokenizer)):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    "\"auto\" tool choice requires "
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )

            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]

            (
                conversation,
                request_prompts,
                engine_prompts,
            ) = await self._preprocess_chat(
                request,
                tokenizer,
                request.messages,
                chat_template=request.chat_template or self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
                tool_dicts=tool_dicts,
                documents=request.documents,
                chat_template_kwargs=request.chat_template_kwargs,
                tool_parser=tool_parser,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
                add_special_tokens=request.add_special_tokens,
            )
        except (ValueError, TypeError, RuntimeError,
                jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # [DEBUG] Log rendered prompt to diagnose model quality issues
        if engine_prompts:
            ep0 = engine_prompts[0]
            if isinstance(ep0, dict):
                _p = ep0.get('prompt')
                _ids = ep0.get('prompt_token_ids')
            else:
                _p = getattr(ep0, 'prompt', None)
                _ids = getattr(ep0, 'prompt_token_ids', None)
            if _p:
                logger.info("[DEBUG_PROMPT] text=%r", _p[:500])
            if _ids:
                logger.info("[DEBUG_PROMPT] token_ids len=%d last20=%s",
                            len(_ids), _ids[-20:])

        # tool_choice = "required" is not supported on BI-V100
        if request.tool_choice == "required":
            return self.create_error_response(
                "tool_choice = \"required\" is not supported!")

        request_id = "chatcmpl-" \
                     f"{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: Union[SamplingParams, BeamSearchParams]

                # [BI100] Adaptive max_tokens based on prompt length.
                _prompt_len = len(engine_prompt["prompt_token_ids"])
                if _prompt_len > 65536:
                    _adaptive_cap = 256
                elif _prompt_len > 32768:
                    _adaptive_cap = 512
                elif _prompt_len > 16384:
                    _adaptive_cap = 1024
                else:
                    _adaptive_cap = 8192

                # OpenAI API: max_completion_tokens takes precedence
                if request.max_completion_tokens is not None \
                        and request.max_tokens is None:
                    request.max_tokens = request.max_completion_tokens

                if request.max_tokens is not None:
                    request.max_tokens = min(request.max_tokens, _adaptive_cap)
                else:
                    request.max_tokens = _adaptive_cap

                default_max_tokens = min(
                    self.max_model_len - _prompt_len, _adaptive_cap)

                # [BI100] Qwen3 thinking-mode sampling fix: greedy / low
                # temperature causes degenerate repeats after empty <think>.
                # Official recommendation: temperature >= 0.6 when thinking.
                # Also set min_tokens to prevent premature EOS right after
                # <think> — the BI-V100 fp16 precision causes <|im_end|>
                # to rank abnormally high in the logits distribution.
                _thinking_on = (
                    request.chat_template_kwargs is not None
                    and request.chat_template_kwargs.get(
                        "enable_thinking", False))
                if _thinking_on:
                    if (request.temperature is None
                            or request.temperature < 0.6):
                        logger.info(
                            "[BI100] Forcing temperature 0.6 (was %s) for "
                            "thinking mode", request.temperature)
                        request.temperature = 0.6
                    if request.top_p is None or request.top_p > 0.95:
                        request.top_p = 0.95
                    if request.min_tokens == 0:
                        request.min_tokens = 16
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        default_max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        default_max_tokens,
                        self.model_config.logits_processor_pattern,
                        self.default_sampling_params)

                logger.info(
                    "[BI100_DEBUG] sampling: temp=%.2f top_p=%.2f "
                    "max_tokens=%s thinking=%s",
                    sampling_params.temperature, sampling_params.top_p,
                    sampling_params.max_tokens, _thinking_on)

                self._log_inputs(request_id,
                                 request_prompts[i],
                                 params=sampling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                    )
                else:
                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        prompt_adapter_request=prompt_adapter_request,
                        priority=request.priority,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert len(generators) == 1
        result_generator, = generators

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id, model_name,
                conversation, tokenizer, request_metadata,
                raw_request=raw_request)

        try:
            return await self.chat_completion_full_generator(
                request, result_generator, request_id, model_name,
                conversation, tokenizer, request_metadata,
                raw_request=raw_request)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    async def _create_sequential_greedy_fanout(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request],
        fanout_count: int,
    ) -> Union[ChatCompletionResponse, ErrorResponse]:
        from vllm.utils import random_uuid as _random_uuid
        request_id = f"chat-{_random_uuid()}"
        created_time = int(time.time())
        responses: list[ChatCompletionResponse] = []
        for _ in range(fanout_count):
            child_request = request.model_copy(deep=True, update={"n": 1})
            child_response = await self.create_chat_completion(
                child_request, raw_request)
            if isinstance(child_response, ErrorResponse):
                return child_response
            if not isinstance(child_response, ChatCompletionResponse):
                logger.error(
                    "Sequential greedy fan-out unexpectedly returned a stream")
                return self.create_error_response(
                    "Failed to aggregate deterministic n=2 completion")
            responses.append(child_response)
        try:
            response = _merge_sequential_chat_responses(
                responses, request_id, created_time)
        except ValueError as error:
            logger.error("Sequential greedy fan-out aggregation failed: %s",
                         type(error).__name__)
            return self.create_error_response(
                "Failed to aggregate deterministic n=2 completion")
        if raw_request is not None:
            metadata = RequestResponseMetadata(
                request_id=request_id,
                final_usage_info=response.usage)
            raw_request.state.request_metadata = metadata
        logger.info("[BI100 N_FANOUT] choices=%d mode=sequential_greedy",
                    fanout_count)
        return response

    @staticmethod
    def _bracket_level(s: str, opening='{', closing='}') -> int:
        """
        Calculate the current level of nested brackets in a given string.
        """
        level = 0
        for char in s:
            if char == opening:
                level += 1
            elif char == closing:
                level -= 1
        return level

    @staticmethod
    def _filter_delta_text(delta_text: str,
                           previous_text: str) -> tuple[str, bool]:
        # remove last '},' of the tool definition stemming from the
        # "name"/"parameters" outer object or closing ']' of the tool list
        # count occurrences of opening and closing curly braces and
        # once level 0 is reached stop outputting text
        # if 0 is reached while parsing the delta_text we know the current
        # tool will finish in this current iteration
        bracket_level = OpenAIServingChat._bracket_level(previous_text)
        updated_delta, passed_zero = "", False
        for c in delta_text:
            if c == '{':
                bracket_level += 1
                passed_zero = bracket_level == 0
            elif c == '}':
                bracket_level -= 1
                passed_zero = bracket_level == 0

            if bracket_level != 0:
                updated_delta += c
            else:
                # if a comma is reached at level 0 we can stop
                if c == ',':
                    break
        return updated_delta, passed_zero

    def extract_tool_call_required_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        function_name_returned: bool,
    ) -> tuple[Optional[DeltaMessage], bool]:
        try:
            obj = partial_json_parser.loads(current_text)
        except partial_json_parser.core.exceptions.MalformedJSON:
            logger.debug('not enough tokens to parse into JSON yet')
            obj = None

        # check if the current text is a valid array
        # containing a partial tool calling object
        # if not repeat
        if obj is None or not isinstance(obj, list) or not len(obj) > 0:
            function_name_returned = False
            delta_message = None
        else:
            _, finishes_previous_tool = OpenAIServingChat._filter_delta_text(
                delta_text, previous_text)
            # take the last tool call from the generated list
            current_tool_call = obj[-1]

            # once parameters have been generated the name is complete as well
            if not finishes_previous_tool and ("name" not in current_tool_call
                                               or "parameters"
                                               not in current_tool_call):
                function_name_returned = False
                delta_message = None
            else:
                if not function_name_returned:
                    # get partly generated arguments from the latest tool call
                    param_match = re.search(r'.*"parameters":\s*(.*)',
                                            current_text)
                    arguments = param_match.group(1) if param_match else ""
                    arguments, _ = OpenAIServingChat._filter_delta_text(
                        arguments, previous_text)

                    # if this iteration finishes a previous tool call but a
                    # new incomplete tool is already generated, take the
                    # previous from the list
                    if (finishes_previous_tool
                            and "parameters" not in current_tool_call):
                        current_tool_call = obj[-2]

                    function_name_returned = True
                    delta_message = DeltaMessage(tool_calls=[
                        DeltaToolCall(function=DeltaFunctionCall(
                            name=current_tool_call["name"],
                            arguments=arguments),
                                      index=len(obj) - 1,
                                      type="function")
                    ])

                else:
                    delta_text, _ = OpenAIServingChat._filter_delta_text(
                        delta_text, previous_text)

                    if delta_text != "":
                        delta_message = DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                function=DeltaFunctionCall(
                                    # OpenAI API returns None
                                    # instead of name every time
                                    name=None,
                                    arguments=delta_text),
                                index=len(obj) - 1,
                                type="function")
                        ])
                    else:
                        delta_message = None

        return delta_message, function_name_returned

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        raw_request: Optional[Request] = None,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request))

        should_stream_with_reasoning_parsing = (
            self._should_stream_with_reasoning_parsing(request))

        all_previous_token_ids: Optional[list[list[int]]]
        function_name_returned: Optional[list[bool]] = None

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if tool_choice_auto or should_stream_with_reasoning_parsing:
            # These are only required in "auto" tool choice case
            previous_texts = [""] * num_choices
            all_previous_token_ids = [[]] * num_choices
            # For reasoning parser and tool call all enabled
            added_content_delta_arr = [False] * num_choices
            reasoning_end_arr = [False] * num_choices
        elif request.tool_choice == "required":
            previous_texts = [""] * num_choices
            function_name_returned = [False] * num_choices
            all_previous_token_ids = None
        else:
            previous_texts, all_previous_token_ids = None, None

        try:
            # There is no need to check if the reasoning_parser is None
            # because the should_stream_with_reasoning_parsing check
            # already ensures that the reasoning_parser is not None.
            # but the pre-commit hook requires it.
            if should_stream_with_reasoning_parsing and \
                self.reasoning_parser is not None:
                reasoning_parser = self.reasoning_parser(tokenizer)
        except RuntimeError as e:
            logger.exception("Error in reasoning parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Prepare the tool parser if it's needed
        try:
            if tool_choice_auto and self.tool_parser:
                tool_parsers: list[Optional[ToolParser]] = [
                    self.tool_parser(tokenizer)
                ] * num_choices
            else:
                tool_parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in tool parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        if stream_options:
            include_usage = stream_options.include_usage
            include_continuous_usage = include_usage and \
                                       stream_options.continuous_usage_stats
        else:
            include_usage, include_continuous_usage = False, False

        # [BI100] Named tool call tracking
        named_tool_call_ids = (
            [f"chatcmpl-tool-{random_uuid()}" for _ in range(num_choices)]
            if tool_choice_function_name else [])
        named_tool_header_sent = [False] * num_choices

        # [BI100] Reasoning token count tracking
        reasoning_token_counts: list[int] = [0] * num_choices

        # [BI100] Disconnect watcher
        _disconnect_watcher: Optional[asyncio.Task] = None
        if raw_request is not None:
            async def _watch_disconnect() -> None:
                try:
                    while True:
                        if await raw_request.is_disconnected():
                            logger.info(
                                "Client disconnected (decode watcher), "
                                "aborting request %s", request_id)
                            await self.engine_client.abort(request_id)
                            return
                        await asyncio.sleep(0.3)
                except asyncio.CancelledError:
                    pass
            _disconnect_watcher = asyncio.ensure_future(_watch_disconnect())

        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens)

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: Union[str, list[dict[str, str]]] = ""
                        if conversation and "content" in conversation[
                                -1] and conversation[-1].get("role") == role:
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        logprobs=None,
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name)
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens)

                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    tool_parser = tool_parsers[i]

                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            return_as_token_id=request.
                            return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    delta_text = output.text

                    if not delta_text and not output.token_ids and \
                        not previous_num_tokens[i]:
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: Optional[DeltaMessage]

                    # just update previous_texts and previous_token_ids
                    if tool_choice_auto or should_stream_with_reasoning_parsing:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text
                        current_token_ids = previous_token_ids + list(
                            output.token_ids)

                    # handle streaming deltas for tools with named tool_choice
                    if tool_choice_function_name:
                        if (self.enable_reasoning
                                and not reasoning_parser.is_reasoning_end(
                                    previous_token_ids)):
                            assert reasoning_parser is not None
                            delta_message = (
                                reasoning_parser.
                                extract_reasoning_content_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                ))
                            # When encountering think end id in delta_token_ids,
                            # process the `content`. Only keep 'content',
                            # remove 'reasoning_content'
                            if reasoning_parser.is_reasoning_end(
                                    list(output.token_ids)):
                                if delta_message and delta_message.content:
                                    # This need to be added to next `delta_text`
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""
                        else:
                            # Just to add remaining `content`
                            if self.enable_reasoning:
                                delta_text = previous_text + delta_text
                                current_text = ""

                            first_named_delta = _consume_named_tool_header_slot(
                                named_tool_header_sent, i)
                            delta_message = DeltaMessage(tool_calls=[
                                DeltaToolCall(**_named_tool_delta_payload(
                                    tool_choice_function_name,
                                    delta_text,
                                    i,
                                    named_tool_call_ids[i],
                                    first_named_delta,
                                ))
                            ])

                    elif request.tool_choice == "required":
                        assert previous_texts is not None
                        assert function_name_returned is not None
                        previous_text = previous_texts[i]
                        current_text = previous_text + delta_text
                        fn_name_returned = function_name_returned[i]

                        delta_message, function_name_returned[i] = (
                            self.extract_tool_call_required_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                function_name_returned=fn_name_returned))

                        # update the previous values for the next iteration
                        previous_texts[i] = current_text

                    # handle streaming deltas for tools with "auto" tool choice
                    # and reasoning parser
                    elif tool_choice_auto and self.enable_reasoning:
                        assert tool_parser is not None
                        assert reasoning_parser is not None
                        assert added_content_delta_arr is not None
                        assert reasoning_end_arr is not None
                        if not reasoning_end_arr[i]:
                            delta_message = (
                                reasoning_parser.
                                extract_reasoning_content_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                ))

                            # When encountering think end id in delta_token_ids,
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning_content'.
                            if reasoning_parser.is_reasoning_end(
                                    list(output.token_ids)):
                                reasoning_end_arr[i] = True
                                current_token_ids =  \
                                    reasoning_parser.extract_content_ids(
                                        list(output.token_ids))
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""

                        # handle tool calls only after reasoning is done,
                        else:
                            delta_token_ids = list(output.token_ids)
                            # First time to tool call,
                            # add the remaining text and token ids
                            # to delta from previous
                            if not added_content_delta_arr[i]:
                                added_content_delta_arr[i] = True
                                previous_text = ""
                                previous_token_ids = []
                                delta_text = current_text
                                delta_token_ids = current_token_ids

                            delta_message = (
                                tool_parser.extract_tool_calls_streaming(
                                    previous_text=previous_text,
                                    current_text=current_text,
                                    delta_text=delta_text,
                                    previous_token_ids=previous_token_ids,
                                    current_token_ids=current_token_ids,
                                    delta_token_ids=delta_token_ids,
                                    request=request))
                    # when only tool calls
                    elif tool_choice_auto:
                        assert tool_parser is not None
                        delta_message = (
                            tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=output.token_ids,
                                request=request))
                    # when only reasoning
                    elif self.enable_reasoning:
                        assert reasoning_parser is not None
                        delta_message = (reasoning_parser.
                                         extract_reasoning_content_streaming(
                                             previous_text,
                                             current_text,
                                             delta_text,
                                             previous_token_ids,
                                             current_token_ids,
                                             output.token_ids,
                                         ))
                    # handle streaming just a content delta
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # update the previous values for the next iteration
                    if tool_choice_auto or should_stream_with_reasoning_parsing:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        continue

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None)

                    # if the model is finished generating
                    else:
                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using guided decoding
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(
                                tool_parser.prev_tool_call_arr) > 0
                            index = len(tool_parser.prev_tool_call_arr
                                        ) - 1 if auto_tools_called else 0
                        else:
                            index = 0

                        if self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output) and tool_parser:
                            latest_delta_len = 0
                            if ((isinstance(
                                    delta_message.tool_calls[0].function,
                                    DeltaFunctionCall)) and isinstance(
                                        delta_message.tool_calls[0].function.
                                        arguments, str)):
                                latest_delta_len = len(
                                    delta_message.tool_calls[0].function.
                                    arguments)

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON
                            expected_call = json.dumps(
                                tool_parser.prev_tool_call_arr[index].get(
                                    "arguments", {}),
                                ensure_ascii=False)

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[
                                index]
                            if (latest_delta_len > 0):
                                actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream
                            remaining_call = expected_call.replace(
                                actual_call, "", 1)
                            # set that as a delta message
                            delta_message = DeltaMessage(tool_calls=[
                                DeltaToolCall(index=index,
                                              function=DeltaFunctionCall(
                                                  arguments=remaining_call).
                                              model_dump(exclude_none=True))
                            ])

                        # [BI100] Count reasoning tokens at finish time.
                        if should_stream_with_reasoning_parsing \
                                and all_previous_token_ids is not None \
                                and self.reasoning_parser is not None:
                            r_parser = self.reasoning_parser(tokenizer)
                            reasoning_token_counts[i] = \
                                r_parser.count_reasoning_tokens(
                                    all_previous_token_ids[i])

                        # Send the finish response for each request.n only once
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=("tool_calls" if (
                                auto_tools_called or tool_choice_function_name)
                                else output.finish_reason),
                            stop_reason=output.stop_reason)

                        finish_reason_sent[i] = True

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                total_reasoning = sum(reasoning_token_counts) \
                    if should_stream_with_reasoning_parsing else None
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                    reasoning_tokens=total_reasoning,
                    prompt_tokens_details=(
                        PromptTokenUsageInfo(cached_tokens=num_cached_tokens)
                        if num_cached_tokens is not None else None),
                )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            total_reasoning = sum(reasoning_token_counts) \
                if should_stream_with_reasoning_parsing else None
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
                reasoning_tokens=total_reasoning)

        except asyncio.CancelledError:
            await self.engine_client.abort(request_id)
            return
        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        finally:
            if _disconnect_watcher is not None and not _disconnect_watcher.done():
                _disconnect_watcher.cancel()
                try:
                    await _disconnect_watcher
                except asyncio.CancelledError:
                    pass
            await self.engine_client.abort(request_id)
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        raw_request: Optional[Request] = None,
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        should_stream_with_reasoning_parsing = (
            self._should_stream_with_reasoning_parsing(request))

        # [BI100] Disconnect watcher for non-streaming
        _disconnect_watcher: Optional[asyncio.Task] = None
        if raw_request is not None:
            async def _watch_disconnect() -> None:
                try:
                    while True:
                        if await raw_request.is_disconnected():
                            logger.info(
                                "Client disconnected (non-stream watcher), "
                                "aborting request %s", request_id)
                            await self.engine_client.abort(request_id)
                            return
                        await asyncio.sleep(0.3)
                except asyncio.CancelledError:
                    pass
            _disconnect_watcher = asyncio.ensure_future(_watch_disconnect())

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            await self.engine_client.abort(request_id)
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))
        finally:
            if _disconnect_watcher is not None and not _disconnect_watcher.done():
                _disconnect_watcher.cancel()
                try:
                    await _disconnect_watcher
                except asyncio.CancelledError:
                    pass
            await self.engine_client.abort(request_id)

        assert final_res is not None

        choices: list[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            # Extract reasoning content if parser is configured.
            reasoning_content: Optional[str] = None
            output_text: str = output.text
            if should_stream_with_reasoning_parsing and \
                self.reasoning_parser is not None:
                try:
                    reasoning_parser = self.reasoning_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in reasoning parser creation.")
                    return self.create_error_response(str(e))
                logger.info(
                    "[DEBUG_REASONING] raw output.text=%r token_ids=%s",
                    output.text[:200] if output.text else output.text,
                    list(output.token_ids[:20]) if output.token_ids else [])
                reasoning_content, extracted = (
                    reasoning_parser.extract_reasoning_content(
                        output.text, request=request))
                logger.info(
                    "[DEBUG_REASONING] after parse: reasoning=%r content=%r",
                    reasoning_content[:200] if reasoning_content else reasoning_content,
                    extracted[:200] if extracted else extracted)
                output_text = extracted or ""
                if isinstance(request.tool_choice,
                              ChatCompletionNamedToolChoiceParam):
                    reasoning_content, output_text = \
                        _reclassify_named_guided_json(
                            reasoning_content, output_text)
            else:
                reasoning_content = None
                output_text = output.text

            named_tool_called = False
            auto_tools_called = False

            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            if (not self.enable_auto_tools or not self.tool_parser) and \
                (not isinstance(request.tool_choice,
                                ChatCompletionNamedToolChoiceParam
                                ) and request.tool_choice != "required"):
                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=output_text)

            # if the request uses tools and specified a tool choice
            elif request.tool_choice and type(
                    request.tool_choice) is ChatCompletionNamedToolChoiceParam:

                named_tool_called = True
                parsed_named_tool_calls: Optional[list[ToolCall]] = None
                if (not _tool_arguments_are_json_object(output_text)
                        and self.tool_parser is not None):
                    try:
                        named_tool_info = self.tool_parser(
                            tokenizer).extract_tool_calls(
                                output_text, request=request)
                        if named_tool_info.tools_called:
                            parsed_named_tool_calls = named_tool_info.tool_calls
                    except RuntimeError as e:
                        logger.warning(
                            "Named tool parser unavailable; preserving raw "
                            "arguments: %s", type(e).__name__)
                named_arguments = _select_named_tool_arguments(
                    output_text,
                    request.tool_choice.function.name,
                    parsed_named_tool_calls,
                )

                tool_call_class = MistralToolCall if isinstance(
                    tokenizer, MistralTokenizer) else ToolCall
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content="",
                    tool_calls=[
                        tool_call_class(function=FunctionCall(
                            name=request.tool_choice.function.name,
                            arguments=named_arguments))
                    ])

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":

                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=output_text)

            # handle when there are tools and tool choice is auto
            elif request.tools and (
                    request.tool_choice == "auto"
                    or request.tool_choice is None) and self.enable_auto_tools \
                    and self.tool_parser:

                try:
                    tool_parser = self.tool_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in tool parser creation.")
                    return self.create_error_response(str(e))

                tool_call_info = tool_parser.extract_tool_calls(
                    output_text if output_text is not None else "",
                    request=request)
                auto_tools_called = tool_call_info.tools_called
                if tool_call_info.tools_called:
                    message = ChatMessage(role=role,
                                          reasoning_content=reasoning_content,
                                          content=tool_call_info.content,
                                          tool_calls=tool_call_info.tool_calls)
                else:
                    message = ChatMessage(role=role,
                                          reasoning_content=reasoning_content,
                                          content=output_text)

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion.")
                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=output_text)

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls" if (
                    auto_tools_called or named_tool_called) else
                output.finish_reason if output.finish_reason else "stop",
                stop_reason=output.stop_reason)
            choices.append(choice_data)

        if request.echo:
            last_msg_content: Union[str, list[dict[str, str]]] = ""
            if conversation and "content" in conversation[-1] and conversation[
                    -1].get("role") == role:
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg['text']
                                             for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content
                                                   or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)

        # [BI100] Reasoning token counting
        total_reasoning_tokens: Optional[int] = None
        if self.reasoning_parser is not None:
            rp = self.reasoning_parser(tokenizer)
            total_reasoning_tokens = sum(
                rp.count_reasoning_tokens(list(output.token_ids))
                for output in final_res.outputs)
        num_cached_tokens = (final_res.num_cached_tokens
                             if hasattr(final_res, 'num_cached_tokens')
                             else None)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            reasoning_tokens=total_reasoning_tokens,
            prompt_tokens_details=(
                PromptTokenUsageInfo(cached_tokens=num_cached_tokens)
                if num_cached_tokens is not None else None),
        )

        request_metadata.final_usage_info = usage

        # [BI100] Sampled prompt logprobs
        prompt_logprobs = final_res.prompt_logprobs
        sample_positions = request.bi100_prompt_logprobs_sample_positions
        if sample_positions is not None:
            if num_cached_tokens not in (None, 0):
                return self.create_error_response(
                    "BI100 sampled prompt logprobs require a cold request.")
            if prompt_logprobs is None or (
                    sample_positions
                    and sample_positions[-1] >= len(prompt_logprobs)):
                return self.create_error_response(
                    "BI100 prompt-logprob sample positions exceed the prompt.")
            selected = set(sample_positions)
            prompt_logprobs = [
                row if position in selected else None
                for position, row in enumerate(prompt_logprobs)
            ]

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(prompt_logprobs),
        )

        return response

    def _get_top_logprobs(
            self, logprobs: dict[int, Logprob], top_logprobs: Optional[int],
            tokenizer: AnyTokenizer,
            should_return_as_token_id: bool) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(token=(token := self._get_decoded_token(
                p[1],
                p[0],
                tokenizer,
                return_as_token_id=should_return_as_token_id)),
                                  logprob=max(p[1].logprob, -9999.0),
                                  bytes=list(
                                      token.encode("utf-8", errors="replace")))
            for i, p in enumerate(logprobs.items())
            if top_logprobs and i < top_logprobs
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[dict[int, Logprob]]],
        tokenizer: AnyTokenizer,
        num_output_top_logprobs: Optional[int] = None,
        return_as_token_id: Optional[bool] = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = return_as_token_id if \
            return_as_token_id is not None else self.return_tokens_as_token_ids
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = tokenizer.decode(token_id)
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    ))
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=None if step_decoded is None else list(
                            step_decoded.encode("utf-8", errors="replace")),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs, num_output_top_logprobs,
                            tokenizer, should_return_as_token_id),
                    ))

        return ChatCompletionLogProbs(content=logprobs_content)

    def _should_stream_with_auto_tool_parsing(self,
                                              request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (request.tools and self.tool_parser and self.enable_auto_tools
                and request.tool_choice in ['auto', None])

    def _should_stream_with_reasoning_parsing(self,
                                              request: ChatCompletionRequest):
        """
            Utility function to check if streamed tokens should go through the
            reasoning parser that was configured.
    
            We only want to do this IF reasoning is enabled and a reasoning 
            parser is configured.
            """
        return self.enable_reasoning and self.reasoning_parser is not None

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: Optional[DeltaMessage],
        output: CompletionOutput,
    ) -> bool:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """

        # yapf: disable
        return bool(
            # if there is a delta message that includes tool calls which
            # include a function that has arguments
            output.finish_reason is not None
            and self.enable_auto_tools and self.tool_parser and delta_message
            and delta_message.tool_calls and delta_message.tool_calls[0]
            and delta_message.tool_calls[0].function
            and delta_message.tool_calls[0].function.arguments is not None
        )
