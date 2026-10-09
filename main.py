
import json
import os
import re
import sys
import traceback
from contextlib import redirect_stdout
from io import StringIO
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


def execute_python_code(code: str) -> dict:
    output_buffer = StringIO()

    try:
        compiled_code = compile(code, "<string>", "exec")

        with redirect_stdout(output_buffer):
            exec(compiled_code, {"__name__": "__main__"})

        return {
            "success": True,
            "output": output_buffer.getvalue(),
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc(),
        }


def find_error_lines_with_ai(code: str, error_output: str) -> List[int]:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://aipipe.org/openai/v1",
    )

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Identify the line numbers in the submitted Python code "
                    "that caused the error. Use 1-based line numbers. "
                    "Return only valid JSON in this format: "
                    '{"error_lines": [1]}. '
                    "Return an empty list if no submitted-code line can "
                    "be identified. Do not return traceback line numbers "
                    "that refer to Python internals."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Submitted Python code:\n"
                    f"{code}\n\n"
                    "Full traceback:\n"
                    f"{error_output}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    parsed = json.loads(content or "{}")
    valid_lines = []

    for line in parsed.get("error_lines", []):
        if isinstance(line, int) and 1 <= line <= len(code.splitlines()):
            if line not in valid_lines:
                valid_lines.append(line)

    return valid_lines


def fallback_error_lines(code: str, error_output: str) -> List[int]:
    total_lines = len(code.splitlines())

    # Prefer the final traceback frame that refers to submitted code.
    matches = re.findall(r'File "<string>", line (\d+)', error_output)

    for match in reversed(matches):
        line_number = int(match)
        if 1 <= line_number <= total_lines:
            return [line_number]

    # SyntaxError often includes its line number in the exception itself.
    match = re.search(r"line (\d+)", error_output.split("SyntaxError:")[0])
    if "SyntaxError:" in error_output and match:
        line_number = int(match.group(1))
        if 1 <= line_number <= total_lines:
            return [line_number]

    return []


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    error_output = execution["output"]

    try:
        error_lines = find_error_lines_with_ai(
            request.code,
            error_output,
        )
    except Exception:
        error_lines = fallback_error_lines(
            request.code,
            error_output,
        )

    return {
        "error": error_lines,
        "result": error_output,
    }


@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoint": "POST /code-interpreter",
    }
