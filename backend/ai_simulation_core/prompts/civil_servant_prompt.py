"""정책 근거 안에서 공무원 답변 본문을 생성하는 Qwen 프롬프트."""

from __future__ import annotations

import json
from collections.abc import Sequence


def civil_servant_prompt(
    persona: dict,
    policy: dict,
    citizen_complaint: str,
    grounded_response: str,
    validation_feedback: Sequence[str] | None = None,
) -> str:
    """공무원 말투만 Qwen이 생성하고 식별자와 정책 근거는 서버가 관리한다."""

    persona_json = json.dumps(persona, ensure_ascii=False, default=str, indent=2)
    policy_json = json.dumps(policy, ensure_ascii=False, default=str, indent=2)
    feedback = (
        json.dumps(list(validation_feedback), ensure_ascii=False)
        if validation_feedback
        else "없음"
    )
    return f"""
당신은 아래 현직 공무원 페르소나의 차분하고 책임 있는 말투로 시민 민원에 답합니다.
추론 과정, 자기소개, 페르소나 프로필은 출력하지 말고 답변 본문만 생성하세요.

[현직 공무원 페르소나]
{persona_json}

[입력 정책 원문]
{policy_json}

[시민 민원]
{citizen_complaint}

[정책 근거 초안]
{grounded_response}

[직전 검증 오류]
{feedback}

작성 규칙:
1. 한국어 2~4문장으로 민원에 직접 답하세요.
2. 정책 근거 초안이 사실 범위의 상한입니다. 의미를 유지해 자연스럽게 바꿀 수 있지만,
   입력에 없는 수치·금액·날짜·기간·서류·신청 경로·연락처를 만들지 마세요.
3. 입력만으로 알 수 없는 정보는 공식 공고 또는 담당 기관 확인이 필요하다고 표현하세요.
4. 시민의 승인·선정·지급·수급 또는 신청기한 연장을 약속하거나 확정하지 마세요.
5. 민원 점수, 위험도, 정책 집행 평가를 만들지 마세요.
6. 직전 검증 오류가 있으면 같은 위반을 반복하지 마세요.
7. 아래 JSON 객체만 출력하세요. 마크다운 코드 블록과 추가 설명은 쓰지 마세요.

{{"response": "정책 원문에 근거한 공무원 답변 본문"}}
""".strip()
