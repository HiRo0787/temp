#!/usr/bin/env python3
"""
Custom Python JSON Tool Parser for vLLM

This parser extracts JSON tool calls from text output, specifically from <tools> tags.
It's designed to parse tool calls that are embedded in the model's text response
and move them into the tool_calls field for proper API compatibility.

The parser looks for:
1. <tools>...</tools> XML tags containing JSON
2. Direct JSON arrays in the text
3. JSON in code blocks (```json ... ```)

This enables tools like Inspect AI to properly detect and use tool calls.
"""

import json
import re
from typing import Sequence, List, Dict, Any, Optional
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ExtractedToolCallInformation,
    DeltaMessage,
)
from transformers import PreTrainedTokenizerBase


class PythonJsonToolParser(ToolParser):
    """
    Custom tool parser that extracts JSON tool calls from text output.
    
    Supports multiple formats:
    - <tools>...</tools> XML tags with JSON
    - Direct JSON arrays
    - JSON in code blocks
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        # Pattern to match <tools>...</tools> tags
        self.tools_tag_pattern = re.compile(
            r'<tools>\s*(.*?)\s*</tools>',
            re.DOTALL | re.IGNORECASE
        )
        # Pattern to match JSON code blocks
        self.json_code_block_pattern = re.compile(
            r'```(?:json)?\s*(.*?)\s*```',
            re.DOTALL | re.IGNORECASE
        )
        # Pattern to match JSON arrays at the start/end of text
        self.json_array_pattern = re.compile(
            r'^\s*(\[.*?\])\s*$',
            re.DOTALL
        )
    
    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Adjust the request if needed (e.g., skip special tokens for tool call output).
        """
        # Don't skip special tokens so we can parse the full output
        return request
    
    def _extract_json_from_text(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract JSON tool calls from various text formats.
        
        Returns a list of tool call dictionaries, or None if no valid JSON found.
        """
        # Try <tools>...</tools> format first (most common for our model)
        tools_match = self.tools_tag_pattern.search(text)
        if tools_match:
            json_str = tools_match.group(1).strip()
            try:
                tool_call_data = json.loads(json_str)
                # Handle both single tool call (dict) and array of tool calls
                if isinstance(tool_call_data, dict):
                    return [tool_call_data]
                elif isinstance(tool_call_data, list):
                    return tool_call_data
            except json.JSONDecodeError:
                pass
        
        # Try JSON code blocks
        code_block_match = self.json_code_block_pattern.search(text)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
            try:
                tool_call_data = json.loads(json_str)
                if isinstance(tool_call_data, dict):
                    return [tool_call_data]
                elif isinstance(tool_call_data, list):
                    return tool_call_data
            except json.JSONDecodeError:
                pass
        
        # Try direct JSON array at start/end of text
        json_array_match = self.json_array_pattern.search(text.strip())
        if json_array_match:
            json_str = json_array_match.group(1)
            try:
                tool_call_data = json.loads(json_str)
                if isinstance(tool_call_data, list):
                    return tool_call_data
            except json.JSONDecodeError:
                pass
        
        # Try parsing the entire text as JSON (fallback)
        try:
            tool_call_data = json.loads(text.strip())
            if isinstance(tool_call_data, dict):
                return [tool_call_data]
            elif isinstance(tool_call_data, list):
                return tool_call_data
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _convert_to_tool_calls(self, tool_call_data: List[Dict[str, Any]], request: ChatCompletionRequest) -> List[Dict[str, Any]]:
        """
        Convert extracted JSON data into vLLM tool call format.
        
        Expected input format:
        {
            "name": "function_name",
            "arguments": {...}
        }
        
        Output format matches OpenAI tool call format:
        {
            "id": "call_...",
            "type": "function",
            "function": {
                "name": "function_name",
                "arguments": "{...}"  # JSON string
            }
        }
        """
        tool_calls = []
        
        for idx, tool_data in enumerate(tool_call_data):
            if not isinstance(tool_data, dict):
                continue
            
            # Extract function name
            function_name = tool_data.get("name") or tool_data.get("function_name")
            if not function_name:
                continue
            
            # Extract arguments
            arguments = tool_data.get("arguments") or tool_data.get("args") or {}
            if not isinstance(arguments, dict):
                # If arguments is a string, try to parse it
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                else:
                    arguments = {}
            
            # Create tool call in OpenAI format
            tool_call = {
                "id": f"call_{idx}_{function_name}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments) if arguments else "{}"
                }
            }
            tool_calls.append(tool_call)
        
        return tool_calls
    
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from the complete model output (non-streaming).
        """
        # Extract JSON from text
        tool_call_data = self._extract_json_from_text(model_output)
        
        if tool_call_data:
            # Convert to tool calls format
            tool_calls = self._convert_to_tool_calls(tool_call_data, request)
            
            if tool_calls:
                # Remove tool call content from the main text
                # Try to remove <tools> tags and their content
                cleaned_content = self.tools_tag_pattern.sub('', model_output).strip()
                # Also remove JSON code blocks if they were used
                cleaned_content = self.json_code_block_pattern.sub('', cleaned_content).strip()
                
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=cleaned_content if cleaned_content else None
                )
        
        # No tool calls found, return original content
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output
        )
    
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Optional[DeltaMessage]:
        """
        Extract tool calls from streaming output (incremental).
        
        For streaming, we check if the current text contains complete tool calls.
        """
        # Extract JSON from current text
        tool_call_data = self._extract_json_from_text(current_text)
        
        if tool_call_data:
            # Convert to tool calls format
            tool_calls = self._convert_to_tool_calls(tool_call_data, request)
            
            if tool_calls:
                # Check if we have new tool calls compared to previous
                prev_tool_call_data = self._extract_json_from_text(previous_text)
                prev_tool_calls = []
                if prev_tool_call_data:
                    prev_tool_calls = self._convert_to_tool_calls(prev_tool_call_data, request)
                
                # Only return delta if we have new tool calls
                if len(tool_calls) > len(prev_tool_calls):
                    new_tool_calls = tool_calls[len(prev_tool_calls):]
                    return DeltaMessage(
                        role="assistant",
                        content=None,
                        tool_calls=new_tool_calls
                    )
        
        # No new tool calls in this delta
        return None


# Register the tool parser (module name matches when loaded via --tool-parser-plugin)
ToolParserManager.register_lazy_module(
    name="python_json",
    module_path="python_json_tool_parser",
    class_name="PythonJsonToolParser",
)
