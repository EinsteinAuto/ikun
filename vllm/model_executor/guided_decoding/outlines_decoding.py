import asyncio
import concurrent.futures
from enum import Enum
from json import dumps as json_dumps
from re import escape as regex_escape
from typing import Tuple, Union

from transformers import PreTrainedTokenizerBase

from vllm.model_executor.guided_decoding.outlines_logits_processors import (
    CFGLogitsProcessor, JSONLogitsProcessor, RegexLogitsProcessor)
from vllm.sampling_params import GuidedDecodingParams


class GuidedDecodingMode(Enum):
    JSON = "json"
    REGEX = "regex"
    CHOICE = "choice"
    GRAMMAR = "grammar"


# https://github.com/outlines-dev/outlines/blob/main/outlines/grammars/json.lark
# the main difference is that we changed the start: value to
# start: object | array, so we are denying scalar values as the root of the
# JSON. Starting with scalars as the root seems to cause llama to generate
# without stop.
JSON_GRAMMAR = r"""
?start: object | array

?value: object
| array
| JSON_STRING
| SIGNED_NUMBER      -> number
| "true"             -> true
| "false"            -> false
| "null"             -> null

array  : "[" _ws [value (_ws "," _ws value)*] _ws "]"
object : "{" _ws [pair (_ws "," _ws pair)*] _ws "}"
pair   : JSON_STRING _ws ":" _ws value
_ws    : JSON_WS?

JSON_STRING: /"(\\["\\/bfnrt]|\\u[0-9a-fA-F]{4}|[^"\\\x00-\x1f])*"/
JSON_WS: /[ \t\r\n]{1,4}/
%import common.SIGNED_NUMBER
"""

global_thread_pool = None  # used for generating logits processor fsm


async def get_outlines_guided_decoding_logits_processor(
    guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizerBase,
    reasoner=None,
) -> Union[JSONLogitsProcessor, RegexLogitsProcessor, CFGLogitsProcessor,
           None]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    """
    global global_thread_pool
    guide, mode = _get_guide_and_mode(guided_params)
    if not guide or not mode:
        return None

    if global_thread_pool is None:
        global_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2)
    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(global_thread_pool,
                                      _get_logits_processor, guide, tokenizer,
                                      mode, guided_params.whitespace_pattern,
                                      reasoner)


def get_local_outlines_guided_decoding_logits_processor(
    guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizerBase,
    reasoner=None,
) -> Union[JSONLogitsProcessor, RegexLogitsProcessor, CFGLogitsProcessor,
           None]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    """
    guide, mode = _get_guide_and_mode(guided_params)
    if not guide or not mode:
        return None

    return _get_logits_processor(guide, tokenizer, mode,
                                 guided_params.whitespace_pattern, reasoner)


def _get_guide_and_mode(
    guided_params: GuidedDecodingParams
) -> Union[Tuple[str, GuidedDecodingMode], Tuple[None, None]]:
    if guided_params.json:
        if isinstance(guided_params.json, dict):
            # turn dict into hashable string
            json = json_dumps(guided_params.json)
        else:
            json = guided_params.json
        return json, GuidedDecodingMode.JSON
    elif guided_params.regex:
        return guided_params.regex, GuidedDecodingMode.REGEX
    elif guided_params.choice:
        # choice just uses regex
        choices = [
            regex_escape(str(choice)) for choice in guided_params.choice
        ]
        choices_regex = "(" + "|".join(choices) + ")"
        return choices_regex, GuidedDecodingMode.CHOICE
    elif guided_params.grammar:
        return guided_params.grammar, GuidedDecodingMode.GRAMMAR
    elif guided_params.json_object:
        return JSON_GRAMMAR, GuidedDecodingMode.GRAMMAR
    else:
        return None, None


def _get_logits_processor(
    guide: str, tokenizer: PreTrainedTokenizerBase, mode: GuidedDecodingMode,
    whitespace_pattern: Union[str, None], reasoner=None,
) -> Union[JSONLogitsProcessor, RegexLogitsProcessor, CFGLogitsProcessor]:
    if mode == GuidedDecodingMode.JSON:
        return JSONLogitsProcessor(guide, tokenizer, whitespace_pattern,
                                   reasoner)
    elif mode == GuidedDecodingMode.REGEX or mode == GuidedDecodingMode.CHOICE:
        return RegexLogitsProcessor(guide, tokenizer, reasoner)
    elif mode == GuidedDecodingMode.GRAMMAR:
        return CFGLogitsProcessor(guide, tokenizer, reasoner)
    else:
        raise ValueError(f"Unknown guided decoding mode {mode}")
