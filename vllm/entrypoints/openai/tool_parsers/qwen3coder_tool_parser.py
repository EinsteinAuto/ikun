import ast
import json
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union

import regex as re

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionToolsParam,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@ToolParserManager.register_module("qwen3_coder")
class Qwen3CoderToolParser(ToolParser):
    """
    Tool parser for Qwen3 models using XML-style tool call format:
      <tool_call><function=name><parameter=key>
      value
      </parameter></function></tool_call>

    Port of vllm-original qwen3coder_tool_parser.py to vllm 0.6.3 API.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: List[Dict] = []
        # Base class uses int; we override with string IDs
        self.current_tool_id: Optional[str] = None  # type: ignore[assignment]
        self.streamed_args_for_tool: List[str] = []

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False

        self._reset_streaming_state()

        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (self.tool_call_start_token_id is None
                or self.tool_call_end_token_id is None):
            raise RuntimeError(
                "Qwen3 XML Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!")

        logger.debug("vLLM Successfully imported tool parser %s !",
                     self.__class__.__name__)


    def _generate_tool_call_id(self) -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self) -> None:
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name: Optional[str] = None
        self.current_param_name: Optional[str] = None
        self.current_param_value: str = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text: str = ""
        self.json_started = False
        self.json_closed = False
        self.accumulated_params: Dict[str, Any] = {}
        self.streaming_request: Optional[ChatCompletionRequest] = None

    def _get_arguments_config(
            self, func_name: str,
            tools: Optional[List[ChatCompletionToolsParam]]) -> Dict:
        if tools is None:
            return {}
        for config in tools:
            if not hasattr(config, "type") or not (
                    hasattr(config, "function")
                    and hasattr(config.function, "name")):
                continue
            if config.type == "function" and config.function.name == func_name:
                if not hasattr(config.function, "parameters"):
                    return {}
                params = config.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        logger.debug("Tool '%s' is not defined in the tools list.", func_name)
        return {}

    def _convert_param_value(self, param_value: str, param_name: str,
                             param_config: Dict, func_name: str) -> Any:
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.debug(
                    "Parsed parameter '%s' is not defined in tool '%s', "
                    "returning string value.", param_name, func_name)
            return param_value

        if (isinstance(param_config[param_name], dict)
                and "type" in param_config[param_name]):
            param_type = str(
                param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"

        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (param_type.startswith("int") or param_type.startswith("uint")
              or param_type.startswith("long")
              or param_type.startswith("short")
              or param_type.startswith("unsigned")):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                v = float(param_value)
                return int(v) if v - int(v) == 0 else v
            except (ValueError, TypeError):
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            lower = param_value.lower()
            if lower not in ["true", "false"]:
                logger.debug(
                    "Parameter '%s' value '%s' is not boolean in tool '%s'.",
                    param_name, param_value, func_name)
            return lower == "true"
        else:
            if (param_type in ["object", "array", "arr"]
                    or param_type.startswith("dict")
                    or param_type.startswith("list")):
                try:
                    return json.loads(param_value)
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.debug(
                        "Could not JSON-decode parameter '%s' for tool '%s'; "
                        "falling back to literal evaluation.",
                        param_name,
                        func_name,
                        exc_info=True)
            try:
                return ast.literal_eval(param_value)
            except (ValueError, SyntaxError, TypeError):
                logger.debug(
                    "Could not literal-eval parameter '%s' for tool '%s'; "
                    "returning string value.",
                    param_name,
                    func_name,
                    exc_info=True)
            return param_value

    def _parse_xml_function_call(
            self, function_call_str: str,
            tools: Optional[List[ChatCompletionToolsParam]]) -> ToolCall:
        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1:]
        param_dict: Dict[str, Any] = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1:])
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]
            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name)
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name,
                arguments=json.dumps(param_dict, ensure_ascii=False)))

    def _get_function_calls(self, model_output: str) -> List[str]:
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [
            match[0] if match[0] else match[1] for match in matched_ranges
        ]
        if not raw_tool_calls:
            raw_tool_calls = [model_output]
        raw_function_calls: List[tuple] = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(
                self.tool_call_function_regex.findall(tool_call))
        return [match[0] if match[0] else match[1]
                for match in raw_function_calls]

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)
        try:
            function_calls = self._get_function_calls(model_output)
            if not function_calls:
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)

            tool_calls = [
                self._parse_xml_function_call(fc, request.tools)
                for fc in function_calls
            ]

            self.prev_tool_call_arr.clear()
            for tc in tool_calls:
                self.prev_tool_call_arr.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })

            content_index = model_output.find(self.tool_call_start_token)
            idx = model_output.find(self.tool_call_prefix)
            content_index = content_index if content_index >= 0 else idx
            content = model_output[:content_index]

            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        if not delta_text:
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text))
                if complete_calls > 0 and self.prev_tool_call_arr:
                    open_calls = (
                        current_text.count(self.tool_call_start_token) -
                        current_text.count(self.tool_call_end_token))
                    if open_calls == 0:
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    return DeltaMessage(content="")
            return None

        self.accumulated_text = current_text

        if self.json_closed and not self.in_function:
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}
                tool_starts = current_text.count(self.tool_call_start_token)
                if self.current_tool_index >= tool_starts:
                    self.is_tool_call_started = False
                return None

        if not self.is_tool_call_started:
            if (self.tool_call_start_token_id in delta_token_ids
                    or self.tool_call_start_token in delta_text):
                self.is_tool_call_started = True
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[:delta_text.index(
                        self.tool_call_start_token)]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                if (current_text.rstrip().endswith(self.tool_call_end_token)
                        and delta_text.strip() == ""):
                    return None
                return DeltaMessage(content=delta_text)

        tool_starts_count = current_text.count(self.tool_call_start_token)
        if self.current_tool_index >= tool_starts_count:
            return None

        # Locate the current tool call's text slice
        tool_start_positions: List[int] = []
        search = 0
        while True:
            search = current_text.find(self.tool_call_start_token, search)
            if search == -1:
                break
            tool_start_positions.append(search)
            search += len(self.tool_call_start_token)

        if self.current_tool_index >= len(tool_start_positions):
            return None

        tool_start_idx = tool_start_positions[self.current_tool_index]
        tool_end_idx = current_text.find(self.tool_call_end_token,
                                         tool_start_idx)
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[tool_start_idx:tool_end_idx +
                                     len(self.tool_call_end_token)]

        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = (tool_text.find(self.tool_call_prefix) +
                              len(self.tool_call_prefix))
                func_end = tool_text.find(">", func_start)
                if func_end != -1:
                    self.current_function_name = tool_text[func_start:func_end]
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True
                    self.prev_tool_call_arr.append({
                        "name": self.current_function_name,
                        "arguments": "{}",
                    })
                    self.streamed_args_for_tool.append("")
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            id=self.current_tool_id,
                            function=DeltaFunctionCall(
                                name=self.current_function_name,
                                arguments=""),
                            type="function",
                        )
                    ])
            return None

        if self.in_function:
            if not self.json_started:
                self.json_started = True
                self.streamed_args_for_tool[self.current_tool_index] += "{"
                return DeltaMessage(tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_index,
                        function=DeltaFunctionCall(arguments="{"),
                    )
                ])

            # Collect all complete parameters in one pass (speculative-decode safe)
            param_starts: List[int] = []
            search = 0
            while True:
                search = tool_text.find(self.parameter_prefix, search)
                if search == -1:
                    break
                param_starts.append(search)
                search += len(self.parameter_prefix)

            json_fragments: List[str] = []
            while not self.in_param and self.param_count < len(param_starts):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" not in remaining:
                    break

                name_end = remaining.find(">")
                current_param_name = remaining[:name_end]
                value_start = param_start + name_end + 1
                value_text = tool_text[value_start:]
                if value_text.startswith("\n"):
                    value_text = value_text[1:]

                param_end_idx = value_text.find(self.parameter_end_token)
                if param_end_idx == -1:
                    next_param = value_text.find(self.parameter_prefix)
                    func_end = value_text.find(self.function_end_token)
                    if next_param != -1 and (func_end == -1
                                              or next_param < func_end):
                        param_end_idx = next_param
                    elif func_end != -1:
                        param_end_idx = func_end
                    else:
                        tool_end_in_value = value_text.find(
                            self.tool_call_end_token)
                        if tool_end_in_value != -1:
                            param_end_idx = tool_end_in_value
                        else:
                            break

                if param_end_idx == -1:
                    break

                param_value = value_text[:param_end_idx]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                self.accumulated_params[current_param_name] = param_value
                param_config = self._get_arguments_config(
                    self.current_function_name or "",
                    self.streaming_request.tools
                    if self.streaming_request else None)
                converted = self._convert_param_value(
                    param_value, current_param_name, param_config,
                    self.current_function_name or "")
                serialized = json.dumps(converted, ensure_ascii=False)

                sep = "" if self.param_count == 0 else ", "
                key = json.dumps(current_param_name, ensure_ascii=False)
                json_fragments.append(f"{sep}{key}: {serialized}")
                self.param_count += 1

            if json_fragments:
                combined = "".join(json_fragments)
                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[
                        self.current_tool_index] += combined
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool))
                return DeltaMessage(tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_index,
                        function=DeltaFunctionCall(arguments=combined),
                    )
                ])

            # Emit closing brace when </function> is seen (after params are done)
            if not self.json_closed and self.function_end_token in tool_text:
                self.json_closed = True
                func_start = (tool_text.find(self.tool_call_prefix) +
                              len(self.tool_call_prefix))
                func_content_end = tool_text.find(self.function_end_token,
                                                   func_start)
                if func_content_end != -1:
                    try:
                        parsed_tool = self._parse_xml_function_call(
                            tool_text[func_start:func_content_end],
                            self.streaming_request.tools
                            if self.streaming_request else None)
                        if self.current_tool_index < len(
                                self.prev_tool_call_arr):
                            self.prev_tool_call_arr[
                                self.current_tool_index]["arguments"] = (
                                    parsed_tool.function.arguments)
                    except Exception:
                        logger.debug("Failed to parse tool call during "
                                     "streaming: %s",
                                     tool_text,
                                     exc_info=True)

                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[
                        self.current_tool_index] += "}"
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool))

                result = DeltaMessage(tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_index,
                        function=DeltaFunctionCall(arguments="}"),
                    )
                ])
                self.in_function = False
                self.accumulated_params = {}
                return result

        return None
