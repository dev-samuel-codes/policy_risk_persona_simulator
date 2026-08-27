"""업로드된 정책 파일(PDF/DOCX/HWPX/TXT/MD)에서 정책 필드를 추출한다.

DB에는 아무것도 저장하지 않는다. 추출 결과는 호출한 쪽(API 응답)으로만 반환되며,
사용자가 검토·수정하고 페르소나를 선택해 시뮬레이션을 제출할 때만
data/runtime/active_policy.json(파일 기반 저장소)에 반영된다.
"""

from __future__ import annotations

import json
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from backend.ai_simulation_core.llm.llm_gateway import run_llm

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB
MAX_EXTRACT_CHARS = 8000  # LLM 프롬프트에 넣을 원문 최대 길이

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".hwpx", ".txt", ".md"}

# 파일 추출 단계가 반환하는 직접 입력 필드. 지역·연령 조건은 검토 화면에서 별도로 입력한다.
POLICY_FIELD_KEYS = (
    "policy_name",
    "target_audience",
    "application_period",
    "effective_date",
    "required_documents",
    "application_method",
    "contact",
    "benefits",
    "exclusion_conditions",
)

FIELD_LABELS = {
    "policy_name": "정책명",
    "target_audience": "지원대상",
    "application_period": "신청기간",
    "effective_date": "시행일",
    "required_documents": "제출서류",
    "application_method": "신청방법",
    "contact": "문의처",
    "benefits": "지원금(혜택)",
    "exclusion_conditions": "제외조건",
}


class PolicyFileExtractionError(Exception):
    """정책 파일에서 정보를 추출하지 못했을 때 발생한다."""


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def _read_hwpx(path: Path) -> str:
    text = ""
    with zipfile.ZipFile(path, "r") as archive:
        for name in archive.namelist():
            if not name.endswith(".xml"):
                continue
            root = ET.fromstring(archive.read(name))
            for node in root.iter():
                if node.text:
                    text += node.text.strip() + "\n"
    return text


def _read_plain_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".hwpx": _read_hwpx,
    ".txt": _read_plain_text,
    ".md": _read_plain_text,
}


def extract_text_from_file(filename: str, content: bytes) -> str:
    """업로드된 파일 바이트에서 원문 텍스트를 추출한다."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise PolicyFileExtractionError(
            f"지원하지 않는 파일 형식입니다 ({extension or '확장자 없음'}). "
            f"지원 형식: {supported}"
        )

    if len(content) > MAX_UPLOAD_BYTES:
        raise PolicyFileExtractionError("파일 용량이 너무 큽니다. (최대 15MB)")

    reader = _READERS[extension]

    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    try:
        text = reader(temp_path)
    except PolicyFileExtractionError:
        raise
    except Exception as error:  # noqa: BLE001 - 파서별 예외를 사용자 메시지로 통일
        raise PolicyFileExtractionError(
            f"파일을 읽는 중 오류가 발생했습니다: {error}"
        ) from error
    finally:
        temp_path.unlink(missing_ok=True)

    text = text.strip()
    if not text:
        raise PolicyFileExtractionError("파일에서 텍스트를 추출하지 못했습니다.")

    return text


def _build_extraction_prompt(text: str) -> str:
    field_lines = "\n".join(
        f'- "{key}" ({FIELD_LABELS[key]})' for key in POLICY_FIELD_KEYS
    )
    truncated = text[:MAX_EXTRACT_CHARS]

    return f"""다음은 정책 공고문에서 추출한 원문입니다.

[원문]
{truncated}

[지시사항]
위 원문을 읽고 아래 항목을 JSON 객체 하나로만 정리해서 답변하세요.
{field_lines}

규칙:
- 반드시 유효한 JSON 객체만 출력하세요. 설명, 코드블록 표시(```), 다른 텍스트를 절대 추가하지 마세요.
- 원문에 없는 내용은 추측하지 말고 빈 문자열("")로 두세요.
- 값은 원문에 있는 표현을 최대한 그대로 사용하세요.
- "effective_date"는 YYYY-MM-DD 형식이 확인되면 그 형식으로, 아니면 원문 표현 그대로 적으세요.

JSON:"""


def _parse_llm_json(raw_response: str) -> dict:
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise PolicyFileExtractionError("LLM 응답에서 JSON을 찾지 못했습니다.")

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise PolicyFileExtractionError(
            f"LLM 응답 JSON 파싱에 실패했습니다: {error}"
        ) from error

    if not isinstance(parsed, dict):
        raise PolicyFileExtractionError("LLM 응답이 JSON 객체 형식이 아닙니다.")

    return parsed


def extract_policy_fields(text: str) -> dict:
    """원문 텍스트에서 정책 필드(dict)를 LLM으로 추출한다. DB에 저장하지 않는다."""
    prompt = _build_extraction_prompt(text)
    raw_response = run_llm(prompt)
    parsed = _parse_llm_json(raw_response)

    fields: dict[str, str] = {}
    for key in POLICY_FIELD_KEYS:
        value = parsed.get(key, "")
        fields[key] = str(value).strip() if value is not None else ""

    return fields


def extract_policy_fields_from_file(filename: str, content: bytes) -> dict:
    """업로드 파일 -> 원문 텍스트 -> LLM 필드 추출까지 한 번에 수행한다."""
    text = extract_text_from_file(filename, content)
    return extract_policy_fields(text)
