"""정책·페르소나 사실에 근거한 시민 응답 프롬프트를 만든다."""

import json
import re

from backend.ai_simulation_core.simulations.citizen_quality import (
    MISSING_POLICY_VALUE,
    build_grounding_facts,
    canonical_document_names,
    display_policy_value,
    persona_source_name,
)


_LATIN_PERSONA_TERMS = {
    "PowerPoint": "발표자료",
    "Instagram": "인스타그램",
    "YouTube": "유튜브",
    "Netflix": "넷플릭스",
    "K-pop": "한국 대중음악",
    "MBTI": "성격유형검사",
    "Excel": "엑셀",
    "TikTok": "틱톡",
    "JavaScript": "자바스크립트",
    "Python": "파이썬",
    "C++": "시플러스플러스",
    "C#": "시샵",
    ".NET": "닷넷",
    "HTML": "웹문서언어",
    "CSS": "웹스타일언어",
    "SQL": "데이터베이스질의어",
    "AWS": "아마존웹서비스",
    "API": "응용프로그램인터페이스",
    "CCTV": "폐쇄회로텔레비전",
    "HACCP": "식품안전관리인증",
    "ICT": "정보통신기술",
    "SNS": "사회관계망서비스",
    "DIY": "직접 만들기",
    "PPT": "발표자료",
    "GPS": "위치정보시스템",
    "CPU": "중앙처리장치",
    "CAD": "설계프로그램",
    "OTT": "온라인동영상서비스",
    "AI": "인공지능",
    "IT": "정보기술",
    "PC": "컴퓨터",
    "TV": "텔레비전",
    "VR": "가상현실",
    "AR": "증강현실",
}


def _koreanize_persona_text(value: object) -> str:
    """모델이 복사하기 쉬운 페르소나 설명의 영문 표기를 한국어로 바꾼다."""
    text = display_policy_value(value)
    for source, replacement in sorted(
        _LATIN_PERSONA_TERMS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])",
            replacement,
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"(?<![A-Za-z0-9])\.?[A-Za-z][A-Za-z0-9.+#/-]*", "", text)
    text = re.sub(r"\(\s*\)|\[\s*\]|\{\s*\}", "", text)
    text = re.sub(r"(^|\s)[./|+#-]+(?=\s|$)", r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?:\s*[,.;:!?])+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _policy_region_condition(policy: dict) -> str:
    region_scope = policy.get("region_scope")
    if region_scope == "nationwide":
        return "전국"
    if region_scope == "specific":
        region = [
            str(policy.get("region_province") or "").strip(),
            str(policy.get("region_district") or "").strip(),
        ]
        return " / ".join(value for value in region if value) or "특정 지역(미상)"
    return "구조화 조건 없음(지원대상·선정기준 원문 확인)"


def _policy_age_condition(policy: dict) -> str:
    if "age_min" not in policy and "age_max" not in policy:
        return "구조화 조건 없음(지원대상·선정기준 원문 확인)"
    age_min = policy.get("age_min")
    age_max = policy.get("age_max")
    if age_min is not None and age_max is not None:
        return f"{age_min}세 이상 {age_max}세 이하(경계 포함)"
    if age_min is not None:
        return f"{age_min}세 이상"
    if age_max is not None:
        return f"{age_max}세 이하"
    return "제한 없음"


def _policy_age_basis(policy: dict) -> str:
    age_basis = policy.get("age_basis")
    if age_basis == "dataset_age":
        return "페르소나 데이터셋의 정수 나이"
    return str(age_basis or "구조화 기준 없음")


def _persona_residence(persona: dict) -> str:
    return str(persona.get("district") or persona.get("province") or "").strip()


def _persona_housing_status(persona: dict) -> str:
    source = " ".join(
        str(persona.get(field) or "")
        for field in ("professional_persona", "family_persona", "persona")
    )
    housing_markers = (
        "자가",
        "무주택",
        "월세",
        "전세",
        "임차",
        "주택을 소유",
        "집을 소유",
    )
    if any(marker in source for marker in housing_markers):
        return "위 페르소나 원문에 명시된 사실만 사용"
    return (
        f'{MISSING_POLICY_VALUE}. "저는 주택을 소유하지 않습니다", '
        '"무주택입니다", "자가입니다"라고 쓰거나 소유 여부를 전제로 삼지 말 것'
    )


def _policy_fact_text(policy: dict) -> str:
    detail = policy.get("상세정보")
    if not isinstance(detail, dict):
        detail = {}
    fields = (
        ("서비스명", "서비스명"),
        ("지원대상", "지원대상"),
        ("선정기준", "선정기준"),
        ("지원내용", "지원내용"),
        ("신청방법", "신청방법"),
        ("신청기한", "신청기한"),
        ("구비서류", "구비서류"),
        ("제외조건", "제외조건"),
        ("문의처", "문의처"),
    )
    return "\n".join(
        f"{label}: {display_policy_value(detail.get(key))}"
        for label, key in fields
    )


def _complaint_focus(persona: dict, policy: dict) -> tuple[str, str]:
    detail = policy.get("상세정보")
    if not isinstance(detail, dict):
        detail = {}

    user_fields = (
        ("신청방법", "신청방법"),
        ("신청기한", "신청기한"),
        ("구비서류", "구비서류"),
        ("문의처", "문의처"),
    )
    missing = [
        label
        for label, key in user_fields
        if not str(detail.get(key) or "").strip()
    ]
    seed_text = str(persona.get("uuid") or "")
    seed = sum((index + 1) * ord(char) for index, char in enumerate(seed_text))
    if missing:
        field = missing[seed % len(missing)]
        subject, object_ = {
            "신청방법": ("신청방법이", "신청방법을"),
            "신청기한": ("신청기한이", "신청기한을"),
            "구비서류": ("구비서류가", "구비서류를"),
            "문의처": ("문의처가", "문의처를"),
        }[field]
        return (
            "정보미제공",
            f"{field} 정보가 제공되지 않은 점만 지적. "
            "complaint_text와 dialogue에서 나이·지역·지원 대상·자격·승인·제외 "
            "단어와 조건 충족 설명을 모두 쓰지 말 것. "
            f"안전한 형식: '{subject} 안내되지 않아 확인이 어렵습니다. "
            f"{object_} 어디서 확인할 수 있는지 안내가 필요합니다.'",
        )

    document_names = canonical_document_names(detail.get("구비서류"))
    document_identity = ", ".join(document_names) or "정책에 제공된 구비서류"
    application_method = display_policy_value(detail.get("신청방법"))
    safe_focuses = [
        (
            "지원내용",
            "정책 원문의 지원 금액·횟수·최대 기간을 그대로 보존하고 지원 규모의 "
            "실효성을 평가. 생활비·주거비 부담 같은 주관적 우려는 허용하되, 입력에 "
            "없는 배우자 병원비·자녀 양육비 같은 구체적인 제3자 사실은 만들지 말 것. "
            "안전한 형식: '제공된 지원 "
            "규모가 현실적인 부담을 충분히 줄일 수 있는지 의문입니다.'",
        ),
        ("신청방법", "제공된 신청방법의 접근성과 편의성을 평가"),
        (
            "신청기한",
            "정책에 제공된 신청기간 자체가 준비하기에 충분한지만 평가. 개인이 "
            "이미 준비했거나 준비를 시작하지 못했다는 현재 이력만 만들지 말 것. "
            "직장·가정·공부 맥락의 주관적 시간 부담은 허용. 안전한 형식: "
            "'제공된 신청기간이 준비하기에 "
            "충분한지 의문입니다.'",
        ),
        (
            "구비서류",
            "제공된 구비서류 이름을 인정한 상태에서 "
            f"다음 항목을 모두 직접 언급: {document_identity}. "
            "준비·발급·제출 방식의 부담만 평가. "
            f"정책의 신청방법 값은 '{application_method}'임을 인정하고, 이미 제공된 "
            "온라인·방문 채널의 이용 가능 여부를 모른다고 말하지 말 것. "
            "정책 구비서류 값에 없는 구체적인 발급처·방문·온라인·제출 채널은 "
            "사실처럼 만들지 말고, 세부 안내가 필요하다는 질문으로만 표현. "
            "서류 항목 자체가 없거나 어떤 서류인지 모른다고 말하지 말 것",
        ),
    ]
    return safe_focuses[seed % len(safe_focuses)]


def _feedback_instruction(error: str) -> str:
    if error == "영어 단어 혼용 감지":
        return (
            "직접 작성한 이름·성격·민원 본문·대화 문장의 알파벳을 모두 "
            "삭제하세요. 이미 채워진 요약 값은 바꾸지 마세요. 입력 원문에 "
            "영문 약어가 있어도 그대로 복사하지 말고 한국어로 바꾸거나 "
            "해당 표현을 생략하세요."
        )
    if error.startswith("CONTRADICTED_POLICY_FACT:"):
        field = error.split(":", 2)[1]
        return (
            f"{field}은 정책 사실 블록에 이미 제공되었습니다. "
            f"{field}이 없거나 알 수 없다고 쓰지 말고, 제공된 내용을 그대로 "
            "인정하거나 다른 쟁점으로 바꾸세요."
        )
    if error.startswith("UNSUPPORTED_POLICY_FACT:"):
        field = error.split(":", 2)[1]
        return (
            f"{field}은 정보 없음입니다. 구체적인 방법·날짜·서류를 만들지 말고 "
            "정보가 제공되지 않았다는 질문으로만 바꾸세요."
        )
    if error.startswith("AGE_ELIGIBILITY_CONTRADICTION"):
        return "코드가 판정한 나이 조건을 충족하므로 나이 때문에 제외된다고 쓰지 마세요."
    if error.startswith("REGION_ELIGIBILITY_CONTRADICTION"):
        return "코드가 판정한 지역 조건을 충족하므로 지역 때문에 제외된다고 쓰지 마세요."
    if error.startswith("UNSUPPORTED_NUMERIC_FACT"):
        return "입력에 없는 금액·기간·개수는 삭제하고, 입력에 있는 숫자만 같은 맥락으로 쓰세요."
    if error.startswith("UNSUPPORTED_DATE_FACT"):
        return "입력에 없는 날짜를 삭제하고, 입력에 있는 날짜만 같은 의미로 쓰세요."
    if error.startswith("UNSUPPORTED_APPLICATION_HISTORY"):
        return (
            "실제로 신청·탈락·승인된 경험과 현재까지 준비했거나 못 했다는 개인 "
            "진행 상태를 삭제하고, 정책 기간 자체에 대한 평가나 질문으로 바꾸세요."
        )
    if error.startswith("UNSUPPORTED_CURRENT_OUTCOME"):
        return (
            "현재 지원이 거절되었다는 전제를 삭제하세요. "
            "승인·선정·지급 결과도 확정하지 말고, 실제 결과가 아니라 "
            "'다른 조건이 있는지 확인이 필요하다'처럼 바꾸세요."
        )
    if error.startswith("UNSUPPORTED_OVERALL_ELIGIBILITY"):
        return (
            "민원 두 문장에서 나이·지역·지원 대상·자격 언급과 조건 충족 설명을 "
            "전부 삭제하세요. 지역·나이 충족만으로 전체 지원 자격이나 수급 "
            "가능성을 단정하지 말고, 이번에 지정된 단일 쟁점만 남기세요."
        )
    if error.startswith("UNSUPPORTED_FAMILY_FINANCIAL_FACT"):
        return (
            "페르소나 입력에 없는 배우자·자녀·부모의 병원비·교육비·양육비 같은 "
            "구체적인 제3자 비용 사실은 삭제하세요. 일반적인 생활비 부담 우려로 "
            "바꾸는 것은 허용됩니다."
        )
    if error.startswith("UNSUPPORTED_DOCUMENT_PROCEDURE"):
        return (
            "입력에 없는 서류 발급처·방문·온라인·제출 채널을 삭제하고, "
            "세부 절차 안내가 필요하다는 질문으로 바꾸세요."
        )
    if error.startswith("DOCUMENT_IDENTITY_MISSING"):
        return (
            "정책에 제공된 구비서류 이름을 모두 직접 언급하고, 서류 목록이 "
            "제공되었다는 사실을 인정한 뒤 세부 절차만 질문하세요."
        )
    if error.startswith("PERSONA_FACT_CONTRADICTION"):
        return "페르소나의 기술·직업·주거 사실을 반대로 쓴 부분을 원본 사실과 맞추세요."
    if error.startswith("UNSUPPORTED_PERSONA_FACT"):
        return (
            '"저는 주택을 소유하지 않습니다", "무주택입니다", "자가입니다" 같은 '
            "1인칭 주거 단정을 전부 빼세요. 제외조건은 "
            '"주택 소유 여부에 따른 적용 기준을 확인할 필요가 있습니다"처럼 '
            "개인 상태를 전제하지 않는 질문으로 바꾸세요."
        )
    if error.startswith("PERSONA_SUMMARY_MISMATCH"):
        field = error.split(":", 1)[1]
        return f"persona_summary의 {field} 값을 출력 계약에 채워진 값 그대로 복사하세요."
    if error.startswith("POLICY_QUALIFIER_MISSING"):
        return "지원액·기간을 언급할 때 정책 원문의 '최대', '한도' 같은 범위 표현을 그대로 유지하세요."
    if error.startswith("DEADLINE_CAUSALITY_CONTRADICTION"):
        return "기한이 늦어서 준비 시간이 부족하다는 뒤집힌 인과를 삭제하고, 정책에 제공된 기간 자체만 자연스럽게 평가하세요."
    if error.startswith("GROUNDING_"):
        return "grounding 객체의 키와 값은 출력 계약에 채워진 값을 그대로 복사하세요."
    if error.startswith("COMPLAINT_BASIS_INVALID"):
        return "각 basis는 허용된 열 가지 값 중 실제 쟁점에 맞는 값 하나만 쓰세요."
    if error.startswith("DUPLICATE_COMPLAINT"):
        return "겹치는 민원을 삭제하거나 서로 다른 단일 쟁점으로 다시 작성하세요."
    return error


def citizen_prompt(
    persona: dict,
    policy: dict,
    validation_feedback: list[str] | None = None,
) -> str:
    persona_basic = {
        "uuid": persona.get("uuid"),
        "age": persona.get("age"),
        "occupation": persona.get("occupation"),
        "sex": persona.get("sex"),
        "province": persona.get("province"),
        "district": persona.get("district"),
    }
    grounding = build_grounding_facts(persona, policy)
    detail = policy.get("상세정보")
    if not isinstance(detail, dict):
        detail = {}
    policy_fields = (
        ("서비스명", "서비스명"),
        ("지원대상", "지원대상"),
        ("선정기준", "선정기준"),
        ("지원내용", "지원내용"),
        ("신청방법", "신청방법"),
        ("신청기한", "신청기한"),
        ("구비서류", "구비서류"),
        ("제외조건", "제외조건"),
        ("문의처", "문의처"),
    )
    provided_fields = [label for label, key in policy_fields if str(detail.get(key) or "").strip()]
    missing_fields = [label for label, key in policy_fields if not str(detail.get(key) or "").strip()]
    focus_basis, focus_instruction = _complaint_focus(persona, policy)
    output_contract = {
        "persona_summary": {
            "이름": persona_source_name(persona),
            "직업": str(persona.get("occupation") or ""),
            "성별": str(persona.get("sex") or ""),
            "나이": str(persona.get("age") or ""),
            "거주지": _persona_residence(persona),
        },
        "grounding": grounding,
        "personality": "",
        "complaints": [
            {
                "basis": focus_basis,
                "complaint_text": "",
                "dialogue": "",
            }
        ],
    }

    feedback_text = ""
    if validation_feedback:
        feedback_text = """

[이전 응답 오류 - 모두 고친 뒤 전체 JSON을 새로 작성]
{errors}
""".format(
            errors="\n".join(
                f"- [{error}] {_feedback_instruction(error)}"
                for error in validation_feedback
            )
        )

    return f"""당신은 아래 페르소나 정보를 가진 한국 시민입니다. 정책과 페르소나에 입력된 사실만 근거로 정책 민원을 작성하세요.

[페르소나 기본 사실 - 변경 금지]
{json.dumps(persona_basic, ensure_ascii=False, indent=2)}

[페르소나 전문 역량 사실 - 변경 금지]
{_koreanize_persona_text(persona.get("professional_persona"))}

[페르소나 주거·가족 사실 - 변경 금지]
{_koreanize_persona_text(persona.get("family_persona"))}

[페르소나 종합 설명 - 변경 금지]
{_koreanize_persona_text(persona.get("persona"))}

[개인 주거 사실]
개인 주택 소유 여부: {_persona_housing_status(persona)}

[정책 사실]
{_policy_fact_text(policy)}

[구조화된 정책 조건]
정책 적용 지역: {_policy_region_condition(policy)}
정책 나이 조건: {_policy_age_condition(policy)}
나이 판단 기준: {_policy_age_basis(policy)}

[코드가 판정한 사실 - 변경 금지]
지역 조건 판정: {grounding["region_status"]}
나이 조건 판정: {grounding["age_status"]}
구조화 지역·나이 판정: {grounding["structured_status"]}
전체 수급 자격과 실제 승인 여부: {grounding["overall_eligibility"]}

[민원 쟁점 선택]
- 정보가 제공된 항목: {", ".join(provided_fields) or "없음"}
- 정보 미제공으로 비판 가능한 항목: {", ".join(missing_fields) or "없음"}
- 값이 "{MISSING_POLICY_VALUE}"인 항목은 정보 공백을 민원으로 쓸 수 있습니다.
- 값이 제공된 항목은 부족함·형평성·실효성을 비판할 수 있지만, 정보가 없거나 알 수 없다고 쓰면 안 됩니다.
- 지역·나이 판정이 충족이면 그 사실을 인정하고, 전체 수급 여부는 다른 조건을 알 수 없다는 조건형 질문으로만 표현하세요.

[이번 응답의 유일한 민원 쟁점 - 변경 금지]
- basis: {focus_basis}
- 초점: {focus_instruction}
- 위 쟁점으로 민원 1개만 작성하고, 다른 basis나 쟁점을 추가하지 마세요.

[사실 근거 규칙]
1. 감정과 평가는 주관적이어도 되지만, 사실 전제는 위 정책·페르소나·판정 블록에서 확인할 수 있어야 합니다.
2. 지역·나이 판정은 코드가 판정한 값을 그대로 사용합니다. 충족인 조건 때문에 제외됐다고 주장하지 마세요.
3. 지역·나이 충족은 최종 승인이나 실제 수급을 뜻하지 않습니다. "지원 대상이 맞다", "자격이 있다", "지원받을 수 있다"처럼 전체 자격을 단정하지 말고, 입력되지 않은 다른 조건은 "알 수 없다" 또는 "확인이 필요하다"고 표현하세요.
4. 값이 "{MISSING_POLICY_VALUE}"인 항목은 정보 부재 자체만 지적할 수 있습니다. 신청 채널·기한·서류·문의처를 새로 만들지 마세요.
5. 값이 제공된 항목을 "없다", "모른다", "안내되지 않았다"고 말하지 마세요. 제공된 정보와 제공되지 않은 정보를 한 문장에서 모두 없다고 묶지 마세요.
6. 실제 신청·승인·탈락·반려·선정·지급 경험과 현재 결과는 입력에 없으므로 만들지 마세요. 현재까지 신청 준비를 했거나 못 했다는 개인 진행 상태도 만들지 마세요. 다른 지원을 받았거나 받지 않았다는 개인 수급 이력, "승인될 것이다", "지급받고 있다", "왜 지원이 안 되는지" 같은 전제도 금지하며, 우려나 질문은 조건형으로 표현하세요.
7. 숫자는 입력에 실제로 등장한 숫자와 같은 단위·맥락으로만 사용하고, "최대", "한도", "이상", "이하" 같은 범위 표현도 보존하세요. 지원금과 개인의 실제 월세·소득·재산을 혼동하지 마세요.
8. 페르소나의 이름·직업·기술·주거·가족 사실을 반대로 말하지 마세요. 직업·가족·일정과 일반적인 생활비·주거비 부담을 연결한 주관적 우려는 허용하지만, 페르소나 입력에 없는 배우자 병원비·자녀 양육비·부모 치료비처럼 구체적인 제3자 비용 사실은 만들지 마세요. 특히 개인 주택 소유 여부가 정보 없음이면 소유·비소유·무주택·자가를 1인칭으로 단정하지 말고, 제외조건의 일반적 적용 기준만 질문하세요.
9. 구비서류 이름이 제공되어도 입력에 없는 구체적인 발급처·방문·온라인·제출 채널을 만들지 마세요. 세부 절차는 사실처럼 단정하지 말고 안내가 필요하다는 질문으로만 표현하세요.
10. basis는 출력 계약에 미리 채워진 값을 그대로 복사하고 다른 값으로 바꾸지 마세요.
11. complaints 배열에는 지정된 쟁점의 민원 1개만 작성하고, 대상자 여부 등 전제를 끝까지 일관되게 유지하세요.
12. 직접 작성하는 이름·성격·민원 본문·대화에는 순수 한국어만 사용하고 영어 단어·알파벳을 쓰지 마세요. 입력 원문에 알파벳이 있어도 한국어로 바꾸거나 생략하세요. 이미 채워진 요약 값은 그대로 유지하세요. 문장은 짧고 명확한 일상 구어체로 작성하세요.
13. 이름·나이·직업·성별·거주지는 출력 계약에 채워진 값이 있으면 그대로 사용하세요. 이름이 빈 경우에만 나이·성별에 맞는 흔한 한국 이름을 새로 지으세요.
14. 추론 과정, 설명, 마크다운 없이 최종 JSON 객체만 출력하세요.

[출력 계약]
아래 JSON의 빈 문자열만 실제 내용으로 채우세요. 이미 채워진 값과 키는 바꾸거나 빼지 마세요.
{json.dumps(output_contract, ensure_ascii=False, indent=2)}

- complaints는 정확히 1개인 배열입니다. 항목을 추가하거나 삭제하지 마세요.
- complaint_text는 핵심 논지 한 문장, dialogue는 같은 민원을 실제 말투로 구체화한 문장입니다.
- personality는 입력된 페르소나 사실에만 근거한 완성 문장입니다.
- 모든 빈 문자열을 채우고, 추가 키를 만들지 마세요.{feedback_text}
""".strip()
