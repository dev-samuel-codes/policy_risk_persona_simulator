"""시민 생성 결과의 정책·페르소나 근거 일관성을 검사한다."""

from __future__ import annotations

import json
import re
import unicodedata
from calendar import monthrange
from datetime import date
from decimal import Decimal
from typing import Any

from backend.ai_simulation_core.region_matching import region_matches


MATCHED = "충족"
NOT_MATCHED = "불충족"
UNKNOWN = "판정불가"
MISSING_POLICY_VALUE = "정보 없음(추정 금지)"

ALLOWED_COMPLAINT_BASES = {
    "지원대상",
    "선정기준",
    "지원내용",
    "신청방법",
    "신청기한",
    "구비서류",
    "제외조건",
    "문의처",
    "정보미제공",
    "개인상황",
}

_UNCERTAINTY_PATTERN = re.compile(
    r"알 수 없|모르|명확하지 않|안내(?:가|도)? 없|확인(?:이|할)? 필요|"
    r"인지 궁금|일까 걱정|될까 걱정|가능성|라면|이라면|경우"
)
_EXCLUSION_PATTERN = re.compile(
    r"지원\s*대상(?:자)?(?:에서)?\s*(?:제외|탈락)|"
    r"지원\s*대상(?:자)?(?:이|은|자가|자는)?\s*(?:아닌|아니|아닙|되지 않)|"
    r"대상(?:자)?(?:이|은|자가|자는)?\s*(?:아닌|아니|아닙|되지 않)|"
    r"지원(?:이|을|금)?\s*(?:받을 수 없|받지 못|못 받|안 되|되지 않|불가)|"
    r"청년(?:이|은)?\s*(?:아니|아닙|되지 않)|"
    r"(?:제외|탈락)(?:되|됐|한)"
)
_EXCLUSION_NEGATION_PATTERN = re.compile(
    r"제외(?:되는|된)? (?:것|건)은? 아니|"
    r"탈락(?:하는|한)? (?:것|건)은? 아니|"
    r"대상이 아닌 (?:것|건)은? 아니"
)
_CAUSAL_BRIDGE_BREAK_PATTERN = re.compile(
    r"아니라|지만|으나|반면|대신|그러나|그런데"
)
_ALTERNATE_CAUSE_PATTERN = re.compile(
    r"(?:다른|추가|소득|재산|서류|주택|무주택|선정|제외조건|가구|예산|취업)"
    r"(?:\s*(?:조건|기준|요건|문제|사유|상태))?.{0,8}"
    r"(?:때문|탓|문제(?:로|라서)?|사유|원인|(?:으)?로\s*인해)"
)
_AGE_MISMATCH_PATTERN = re.compile(
    r"(?:나이|연령)(?:가|는|이|은|도|을|를)?\s*"
    r"(?:(?:조건|기준|요건|제한|범위|상한|하한)"
    r"(?:은|는|이|가|을|를|에|에서|상)?\s*)?"
    r"(?:불충족|충족하지 못|맞지 않|해당하지 않|"
    r"다르(?!지(?:는)?\s*않)|아니|아닙|벗어나(?!지(?:는)?\s*않)|"
    r"미달(?!하지(?:는)?\s*않)|초과(?!하지(?:는)?\s*않)|"
    r"범위\s*밖|제한에\s*걸려)|"
    r"청년(?:이|은|도)?\s*(?:아니|아닙|되지 않)|"
    r"청년\s*(?:기준|조건|요건|연령|나이)"
    r"(?:에서|에|은|는|이|가|을|를|상)?\s*"
    r"(?:불충족|충족하지 못|맞지 않|해당하지 않|"
    r"벗어나(?!지(?:는)?\s*않)|미달(?!하지(?:는)?\s*않)|"
    r"초과(?!하지(?:는)?\s*않))"
)
_AGE_DIRECT_CAUSE_PATTERN = re.compile(
    r"(?:나이|연령)(?:가|는|이|은|도|를|을)?"
    r"(?:\s*(?:조건|기준|요건|제한|범위|상한|하한)"
    r"(?:이|가|은|는|을|를|에|에서|상)?)?\s*"
    r"(?:때문(?:에)?|탓(?:에)?|이유로|(?:으)?로\s*인해)|"
    r"(?:나이|연령)(?:가|는|이|은)?\s*(?:어려서|많아서)|"
    r"청년\s*(?:기준|조건|요건|연령|나이)\s*"
    r"(?:때문(?:에)?|탓(?:에)?|이유로|(?:으)?로\s*인해)|"
    r"청년(?:층)?(?:이라서|이라는\s*이유로|이기\s*때문(?:에)?)"
)
_REGION_MISMATCH_PATTERN = re.compile(
    r"(?:지역|거주지|주소지|관할(?:지역)?)(?:가|는|이|은|도|을|를)?\s*"
    r"(?:(?:조건|기준|요건|제한|범위)"
    r"(?:은|는|이|가|을|를|에|에서|상)?\s*)?"
    r"(?:불충족|충족하지 못|맞지 않|해당하지 않|"
    r"다르(?!지(?:는)?\s*않)|아니|아닙|벗어나(?!지(?:는)?\s*않)|"
    r"범위\s*밖|제한에\s*걸려)|"
    r"(?:지역|거주지|주소지|관할(?:지역)?).{0,24}"
    r"(?:일치하지 않|범위.{0,8}(?:맞지 않|"
    r"벗어나(?!지(?:는)?\s*않)|밖))"
)
_REGION_DIRECT_CAUSE_PATTERN = re.compile(
    r"(?:지역|거주지|주소지|관할(?:지역)?)"
    r"(?:\s*(?:조건|기준|요건|제한|범위))?\s*"
    r"(?:때문|탓|문제로|(?:으)?로\s*인해)"
)
_CAUSE_NEGATION_SUFFIX_PATTERN = re.compile(
    r"^\s*(?:이\s*)?아니라|"
    r"^\s*.{0,18}(?:것|게|점|설명|사실)"
    r"(?:이|가|은|는)?\s*(?:아니|틀렸)"
)
_REVERSE_CAUSE_BRIDGE_PATTERN = re.compile(
    r"^(?:.{0,24})(?:이유|원인|사유)(?:은|는|이|가)?\s*$"
)
_CHANNEL_PATTERN = re.compile(
    r"온라인|인터넷|누리집|홈페이지|모바일|애플리케이션|앱|웹|"
    r"방문|주민센터|행정복지센터|전화|이메일|전자우편|우편|팩스"
)
_METHOD_ASSERTION_PATTERN = re.compile(
    r"(?:만|뿐|전용|해야|가능|불가|안 되|되지 않|못 하|없|"
    r"되어 있|라고 되어|통해서|통해)|"
    r"신청\s*(?:시스템|절차|서식).{0,20}(?:복잡|오류|한 번에 안)"
)
_APPLICATION_HISTORY_PATTERN = re.compile(
    r"신청(?:을)?\s*했|지원(?:을)?\s*신청했|"
    r"탈락했|거절당|반려됐|선정됐|지급받았|지원받았|"
    r"(?:지원|수당|보조금).{0,16}(?:받지 않았|받은 적 없|받고 있지 않)|"
    r"(?:지금까지|현재까지).{0,24}(?:신청\s*)?준비(?:를|도)?"
    r"(?:\s*시작(?:조차)?)?\s*"
    r"(?:못|하지\s*못|안\s*했|하지\s*않)"
)
_PERSONAL_PREPARATION_STATE_PATTERN = re.compile(
    r"(?:지금까지|현재까지).{0,24}(?:신청\s*)?준비(?:를|도)?"
    r"(?:\s*시작(?:조차)?)?\s*"
    r"(?:못|하지\s*못|안\s*했|하지\s*않)"
)
_CURRENT_DENIAL_PATTERN = re.compile(
    r"왜.{0,30}(?:지원(?:이|을)?\s*(?:안 되는지|안 됩니다|되지 않는지|못 받는지|못 받습니다)|"
    r"대상(?:에서)?\s*제외되는지)|"
    r"(?:지원(?:이|을)?\s*(?:안 되는지|안 됩니다|되지 않는지|못 받는지|못 받습니다))"
    r".{0,20}(?:왜|이유|모르)"
)
_OVERALL_ELIGIBILITY_ASSERTION_PATTERN = re.compile(
    r"(?:저는|제가|나는|내가).{0,80}(?:지원\s*)?(?:대상(?:자)?|자격).{0,16}"
    r"(?:맞|해당|있|충족|됩니다|입니다)|"
    r"(?:지원\s*)?(?:대상(?:자)?|자격)(?:이|은|에는|에)?\s*"
    r"(?:맞습니다|해당합니다|있습니다|충족합니다)|"
    r"(?:조건|기준).{0,16}충족하니.{0,20}(?:대상|자격|지원받)|"
    r"지원(?:금)?(?:을)?\s*받을\s*수\s*있"
    r"(?:습니다|다|지만|으나|는데|다고|으니)"
)
_CURRENT_OUTCOME_ASSERTION_PATTERN = re.compile(
    r"(?:승인|선정|지급|수급)(?:이|은|을|금이|금은)?\s*"
    r"(?:되었|됐|받았|받고\s*있|확정되었|확정됐|될\s*것)|"
    r"지원금(?:을|이)?\s*(?:받았|받고\s*있|나왔|지급되었|지급됐)"
)
_NUMBER_UNIT_PATTERN = re.compile(
    r"(?<!\d)(\d[\d,]*(?:\.\d+)?)\s*"
    r"(만\s*원|억\s*원|원|세|살|개월|년|월|일|회|퍼센트|%)"
)
_KRW_PREFIX_PATTERN = re.compile(
    r"(?i)(?:KRW|₩)\s*(\d[\d,]*(?:\.\d+)?)"
)
_KRW_SUFFIX_PATTERN = re.compile(
    r"(?i)(?<!\d)(\d[\d,]*(?:\.\d+)?)\s*(?:KRW|₩)"
)
_ENGLISH_NUMBER_UNIT_PATTERN = re.compile(
    r"(?i)(?<!\d)(\d[\d,]*(?:\.\d+)?)\s*"
    r"(months?|years?|times?|percent)\b"
)
_MAXIMUM_NUMBER_UNIT_PATTERN = re.compile(
    r"(?:최대|최고|상한|한도)\s*(\d[\d,]*(?:\.\d+)?)\s*"
    r"(만\s*원|억\s*원|원|개월|년|월|일|회|퍼센트|%)"
)
_ENGLISH_MAXIMUM_NUMBER_UNIT_PATTERN = re.compile(
    r"(?i)up\s+to\s+(\d[\d,]*(?:\.\d+)?)\s*"
    r"(months?|years?|times?|percent)\b"
)
_SOURCE_DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})(?!\d)"
)
_KOREAN_FULL_DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"
)
_KOREAN_MONTH_DAY_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일"
)
_POSITIVE_SKILL_PATTERN = re.compile(
    r"완벽|숙련|전문|능숙|잘 다루|활용 능력|역량을 갖"
)
_NEGATIVE_SKILL_PATTERN = re.compile(
    r"못 다루|못 다뤄|(?:다룰|쓸|할|사용할) 줄 (?:모르|모릅)|"
    r"전혀 모르|문외한|미숙|서툴"
)
_CAPABILITY_KEYWORDS = (
    "엑셀",
    "컴퓨터",
    "디지털",
    "온라인",
    "스마트폰",
    "모바일",
    "정보 시스템",
    "서버",
)
_SELF_RENT_PATTERN = re.compile(
    r"(?:내|제가|나는|제가 살고 있는|내가 살고 있는).{0,35}"
    r"(?:월세|임차료)|(?:월세|임차료)(?:를|가)?\s*(?:내|내고|냅니다)"
)
_DIRECT_FIELD_ABSENCE_PATTERN = re.compile(
    r"^\s*(?:의\s*)?(?:이|가|은|는|도)?\s*"
    r"(?:정보|내용|안내|설명|기준|방법)?\s*"
    r"(?:이|가|은|는|도)?\s*"
    r"(?:(?:무엇|어디|어떤 것)인지\s*)?"
    r"(?:(?:전혀|명확히|정확히)\s*)?"
    r"(?:없|제시되지 않|명시되지 않|안내되지 않|알 수 없|모르)"
)
_GROUP_FIELD_ABSENCE_PATTERN = re.compile(
    r"(?:모두|전부|어떤\s*(?:정보|내용|안내)도|정보\s*없음|안내\s*없음)"
    r".{0,8}(?:없|제시되지 않|명시되지 않|안내되지 않|알 수 없|모르)?"
)
_FIELD_IDENTITY_UNKNOWN_PATTERN = re.compile(
    r"(?:무엇인지|어떤\s*(?:정보|내용|안내|서류)인지|어떤\s*서류를)"
    r".{0,48}(?:없|제시되지 않|명시되지 않|안내되지 않|알 수 없|모르)"
)
_FIELD_AVAILABILITY_ADMISSION_PATTERN = re.compile(
    r"(?:있|제공되|안내되|명시되|기재되|알려져)"
    r".{0,8}(?:지만|으나|는데|반면)"
)
_POLICY_FIELD_ANCHORS = {
    "서비스명": r"서비스\s*명|정책\s*명|사업\s*명",
    "소관기관명": r"소관\s*기관(?:명)?|담당\s*기관|관리\s*기관",
    "지원대상": r"지원\s*대상",
    "선정기준": r"선정\s*기준",
    "지원내용": r"지원\s*내용|지원\s*혜택",
    "신청방법": r"신청\s*방법|신청\s*절차",
    "신청기한": r"신청\s*기한|신청\s*기간|접수\s*기한|접수\s*기간",
    "구비서류": r"구비\s*서류|제출\s*서류|필요(?:한)?\s*서류",
    "제외조건": r"제외\s*조건",
    "문의처": r"문의처|문의\s*기관|연락처",
}
_SUPPORT_AMOUNT_UNKNOWN_PATTERN = re.compile(
    r"지원금(?:이|은|을)?\s*(?:얼마|금액|규모).{0,36}"
    r"(?:모르|알 수 없|안내되지 않|명시되지 않)"
)
_GENERIC_DOCUMENT_UNKNOWN_PATTERN = re.compile(
    r"(?:어떤|무슨)\s*서류(?:를|가|인지|인지도)"
    r".{0,50}(?:없|안내되지 않|알 수 없|모르)"
)
_PERSONAL_HOUSING_ASSERTION_PATTERN = re.compile(
    r"(?:저는|제가|나는|내가|저희는|저희가).{0,35}"
    r"(?:주택|집|아파트|자가|무주택).{0,30}"
    r"(?:소유|보유|자가|무주택)"
    r"|(?:저는|제가|나는|내가|저희는|저희가).{0,35}(?:자가|무주택)"
)
_PERSONA_HOUSING_SOURCE_PATTERN = re.compile(
    r"자가|무주택|월세|전세|임차|"
    r"(?:주택|집|아파트).{0,20}(?:소유|보유)|"
    r"(?:소유|보유).{0,20}(?:주택|집|아파트)"
)
_SPECIFIC_FAMILY_EXPENSE_ASSERTION_PATTERN = re.compile(
    r"(?:아내|남편|배우자|자녀|아이|부모님?|가족).{0,60}"
    r"(?:생활비|주거비|월세|임차료|생계비?|병원비|치료비|약값|의료비|"
    r"교육비|학원비|양육비|보육비|부담|충당|감당)|"
    r"(?:생활비|주거비|월세|임차료|생계비?|병원비|치료비|약값|의료비|"
    r"교육비|학원비|양육비|보육비|부담|충당|감당).{0,60}"
    r"(?:아내|남편|배우자|자녀|아이|부모님?|가족)"
)
_DOCUMENT_PROCEDURE_CHANNEL_PATTERN = re.compile(
    r"온라인|인터넷|누리집|홈페이지|모바일|애플리케이션|앱|웹|"
    r"방문|주민센터|행정복지센터|구청|시청|정부24|무인발급기|"
    r"전화|이메일|전자우편|우편|팩스"
)
_DOCUMENT_PROCEDURE_ASSERTION_PATTERN = re.compile(
    r"(?:해야|하여야|필요|통해|통해서|만\s*가능|밖에|의무|절차|과정|방식)"
)
_DOCUMENT_PROCEDURE_UNCERTAINTY_PATTERN = re.compile(
    r"어떻게|어디(?:에|서)|(?:해야|가능한|되는|제출하는)지|"
    r"안내.{0,16}(?:필요|부족|명확하지|되지\s*않|없)"
)
_DOCUMENT_PROCESS_ANCHOR_PATTERN = re.compile(
    r"서류|신분증|증명서|계약서|발급|제출|준비"
)
_KNOWN_APPLICATION_CHANNEL_PATTERNS = {
    "online": re.compile(
        r"온라인|인터넷|누리집|홈페이지|모바일|애플리케이션|앱|웹|\bonline\b|\bweb\b",
        re.IGNORECASE,
    ),
    "visit": re.compile(
        r"방문|주민센터|행정복지센터|커뮤니티\s*센터|"
        r"\bvisit\b|community\s+(?:service\s+)?center",
        re.IGNORECASE,
    ),
}
_KNOWN_CHANNEL_AVAILABILITY_QUESTION_PATTERNS = {
    "online": re.compile(
        r"(?:온라인|인터넷|누리집|홈페이지|모바일|앱|웹)(?:으로|에서)?"
        r".{0,20}(?:신청|제출).{0,16}(?:할\s*수\s*있는지|가능한지|되는지|가능\s*여부)"
    ),
    "visit": re.compile(
        r"(?:방문|주민센터|행정복지센터|커뮤니티\s*센터).{0,24}"
        r"(?:신청|제출).{0,16}(?:할\s*수\s*있는지|가능한지|되는지|가능\s*여부)"
    ),
}
_DEADLINE_CAUSALITY_CONTRADICTION_PATTERN = re.compile(
    r"(?:신청\s*)?(?:기한|마감).{0,36}(?:늦|뒤).{0,28}"
    r"(?:준비.{0,16}(?:안\s*되|못|어렵)|시간.{0,12}(?:부족|없))"
)
_DOCUMENT_CONCEPTS = (
    (
        ("identification card", "identity card", "id card", "신분증", "주민등록증"),
        ("신분증", "주민등록증", "아이디 카드", "아이디카드"),
        "신분증",
    ),
    (
        (
            "proof of residence",
            "residence certificate",
            "거주 증명서",
            "거주증명서",
            "주민등록등본",
        ),
        ("거주 증명서", "거주증명서", "거주 확인서", "주민등록등본"),
        "거주 증명서",
    ),
    (
        ("income certificate", "proof of income", "소득 증명서", "소득증명서"),
        ("소득 증명서", "소득증명서", "소득 확인서"),
        "소득 증명서",
    ),
    (
        ("bankbook copy", "copy of bankbook", "통장 사본", "통장사본"),
        ("통장 사본", "통장사본"),
        "통장 사본",
    ),
    (
        ("lease agreement", "rental agreement", "임대차 계약서", "임대차계약서"),
        ("임대차 계약서", "임대차계약서", "임대 계약서"),
        "임대차 계약서",
    ),
    (
        ("application form", "신청서"),
        ("신청서",),
        "신청서",
    ),
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", _text(value))
    return re.sub(r"[\s\-_·]", "", normalized).lower()


def required_document_alias_groups(value: Any) -> list[tuple[str, ...]]:
    source = _normalize(value)
    return [
        aliases
        for source_aliases, aliases, _ in _DOCUMENT_CONCEPTS
        if any(_normalize(alias) in source for alias in source_aliases)
    ]


def canonical_document_names(value: Any) -> list[str]:
    source = _normalize(value)
    return [
        canonical
        for source_aliases, _, canonical in _DOCUMENT_CONCEPTS
        if any(_normalize(alias) in source for alias in source_aliases)
    ]


def persona_source_name(persona: dict) -> str:
    """페르소나 원문이 명시한 첫 번째 인물 이름을 반환한다."""
    for field in ("professional_persona", "family_persona", "persona"):
        match = re.search(r"(?<![가-힣])([가-힣]{2,4})\s*씨", _text(persona.get(field)))
        if match:
            return match.group(1)
    return ""


def _normalize_unit(unit: str) -> str:
    return re.sub(r"\s+", "", unit).replace("퍼센트", "%")


def _canonical_number_unit(number: str, unit: str) -> tuple[str, str]:
    normalized_unit = _normalize_unit(unit)
    value = Decimal(number.replace(",", ""))
    if normalized_unit == "만원":
        value *= Decimal(10_000)
        normalized_unit = "원"
    elif normalized_unit == "억원":
        value *= Decimal(100_000_000)
        normalized_unit = "원"

    normalized_number = format(value.normalize(), "f")
    if "." in normalized_number:
        normalized_number = normalized_number.rstrip("0").rstrip(".")
    return normalized_number or "0", normalized_unit


def _english_source_number_units(source_text: str) -> set[tuple[str, str]]:
    allowed = {
        _canonical_number_unit(number, "원")
        for pattern in (_KRW_PREFIX_PATTERN, _KRW_SUFFIX_PATTERN)
        for number in pattern.findall(source_text)
    }
    unit_aliases = {
        "month": "개월",
        "months": "개월",
        "year": "년",
        "years": "년",
        "time": "회",
        "times": "회",
        "percent": "%",
    }
    allowed.update(
        _canonical_number_unit(number, unit_aliases[unit.lower()])
        for number, unit in _ENGLISH_NUMBER_UNIT_PATTERN.findall(source_text)
    )
    return allowed


def _maximum_number_units(value: Any) -> set[tuple[str, str]]:
    source = _text(value)
    maximums = {
        _canonical_number_unit(number, unit)
        for number, unit in _MAXIMUM_NUMBER_UNIT_PATTERN.findall(source)
    }
    unit_aliases = {
        "month": "개월",
        "months": "개월",
        "year": "년",
        "years": "년",
        "time": "회",
        "times": "회",
        "percent": "%",
    }
    maximums.update(
        _canonical_number_unit(number, unit_aliases[unit.lower()])
        for number, unit in _ENGLISH_MAXIMUM_NUMBER_UNIT_PATTERN.findall(source)
    )
    return maximums


def _match_has_maximum_qualifier(sentence: str, match: re.Match[str]) -> bool:
    prefix = sentence[max(0, match.start() - 14) : match.start()]
    suffix = sentence[match.end() : min(len(sentence), match.end() + 6)]
    return bool(
        re.search(r"(?:최대|최고|상한|한도)\s*$", prefix)
        or re.match(r"\s*까지", suffix)
    )


def normalize_maximum_support_qualifiers(result: dict, policy: dict) -> dict:
    """정책의 최대값을 대화체에서 축약한 경우 원문의 범위 표현을 복원한다."""
    maximums = _maximum_number_units(_policy_detail(policy).get("지원내용"))
    complaints = result.get("complaints")
    if not maximums or not isinstance(complaints, list):
        return result
    for complaint in complaints:
        if not isinstance(complaint, dict):
            continue
        for field in ("complaint_text", "dialogue"):
            value = complaint.get(field)
            if not isinstance(value, str):
                continue
            normalized = value
            for match in reversed(list(_NUMBER_UNIT_PATTERN.finditer(value))):
                pair = _canonical_number_unit(*match.groups())
                if pair not in maximums or _match_has_maximum_qualifier(value, match):
                    continue
                normalized = normalized[: match.start()] + "최대 " + normalized[match.start() :]
            complaint[field] = normalized
    return result


def _known_application_channels(value: Any) -> set[str]:
    source = _text(value)
    return {
        channel
        for channel, pattern in _KNOWN_APPLICATION_CHANNEL_PATTERNS.items()
        if pattern.search(source)
    }


def _policy_detail(policy: dict) -> dict:
    detail = policy.get("상세정보")
    return detail if isinstance(detail, dict) else {}


def _claims_field_absence(sentence: str, anchor: str) -> bool:
    """제공 필드 자체의 부재 주장만 잡고 세부 기준 비판은 허용한다."""
    for match in re.finditer(rf"(?:{anchor})", sentence):
        tail = sentence[match.end() : match.end() + 80]
        if _DIRECT_FIELD_ABSENCE_PATTERN.match(tail):
            return True

        grouped = _GROUP_FIELD_ABSENCE_PATTERN.search(tail)
        if grouped:
            segment = tail[: grouped.end()]
            if not _FIELD_AVAILABILITY_ADMISSION_PATTERN.search(segment):
                return True

        identity_unknown = _FIELD_IDENTITY_UNKNOWN_PATTERN.search(tail)
        if identity_unknown:
            segment = tail[: identity_unknown.end()]
            if not _FIELD_AVAILABILITY_ADMISSION_PATTERN.search(segment):
                return True
    return False


def display_policy_value(value: Any) -> str:
    return _text(value) or MISSING_POLICY_VALUE


def evaluate_region_status(persona: dict, policy: dict) -> str:
    scope = _text(policy.get("region_scope"))
    if scope == "nationwide":
        return MATCHED
    if scope != "specific":
        return UNKNOWN

    policy_province = _text(policy.get("region_province"))
    persona_province = _text(persona.get("province"))
    if not policy_province or not persona_province:
        return UNKNOWN
    return (
        MATCHED
        if region_matches(
            persona,
            region_scope=scope,
            province=policy_province,
            district=_text(policy.get("region_district")),
        )
        else NOT_MATCHED
    )


def evaluate_age_status(persona: dict, policy: dict) -> str:
    if "age_min" not in policy and "age_max" not in policy:
        return UNKNOWN

    age = persona.get("age")
    if age is None:
        return UNKNOWN
    try:
        normalized_age = int(age)
    except (TypeError, ValueError):
        return UNKNOWN

    age_min = policy.get("age_min")
    age_max = policy.get("age_max")
    if age_min is not None and normalized_age < int(age_min):
        return NOT_MATCHED
    if age_max is not None and normalized_age > int(age_max):
        return NOT_MATCHED
    return MATCHED


def build_grounding_facts(persona: dict, policy: dict) -> dict:
    region_status = evaluate_region_status(persona, policy)
    age_status = evaluate_age_status(persona, policy)
    if NOT_MATCHED in {region_status, age_status}:
        structured_status = NOT_MATCHED
    elif region_status == MATCHED and age_status == MATCHED:
        structured_status = MATCHED
    else:
        structured_status = UNKNOWN

    return {
        "region_status": region_status,
        "age_status": age_status,
        "structured_status": structured_status,
        "overall_eligibility": "입력만으로 확정 불가",
    }


def _generated_entries(result: dict) -> list[tuple[str, str]]:
    entries = [("personality", _text(result.get("personality")))]
    complaints = result.get("complaints")
    if not isinstance(complaints, list):
        return entries
    for index, complaint in enumerate(complaints):
        if not isinstance(complaint, dict):
            continue
        for field in ("complaint_text", "dialogue"):
            entries.append(
                (f"complaints[{index}].{field}", _text(complaint.get(field)))
            )
    return entries


def _sentences(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    sentences = []
    for path, text in entries:
        for sentence in re.split(r"(?<=[.!?。])\s+|\n+", text):
            sentence = sentence.strip()
            if sentence:
                sentences.append((path, sentence))
    return sentences


def _allowed_number_units(persona: dict, policy: dict) -> set[tuple[str, str]]:
    source_text = json.dumps(
        {"persona": persona, "policy": policy},
        ensure_ascii=False,
        default=str,
    )
    allowed = {
        _canonical_number_unit(number, unit)
        for number, unit in _NUMBER_UNIT_PATTERN.findall(source_text)
    }
    allowed.update(_english_source_number_units(source_text))

    # 입력의 ISO/점/슬래시 날짜를 모델이 자연어 날짜로 바꾸어 쓰는 것은
    # 새로운 숫자 생성이 아니다. 예: 2026-09-30 -> 2026년 9월 30일.
    for year, month, day in _SOURCE_DATE_PATTERN.findall(source_text):
        allowed.add((year, "년"))
        allowed.add((month, "월"))
        allowed.add((day, "일"))
        allowed.add((str(int(month)), "월"))
        allowed.add((str(int(day)), "일"))

    # 월의 첫날부터 마지막 날까지 명시된 신청기간은 같은 의미의 개월 수로
    # 표현할 수 있다. 입력에 없는 개인 일정이 아니라 결정적으로 계산되는 값만 허용한다.
    application_period = _text(_policy_detail(policy).get("신청기한"))
    period_dates = _SOURCE_DATE_PATTERN.findall(application_period)
    if len(period_dates) >= 2:
        start_parts = tuple(int(part) for part in period_dates[0])
        end_parts = tuple(int(part) for part in period_dates[1])
        try:
            start_date = date(*start_parts)
            end_date = date(*end_parts)
        except ValueError:
            pass
        else:
            if (
                start_date <= end_date
                and start_date.day == 1
                and end_date.day == monthrange(end_date.year, end_date.month)[1]
            ):
                calendar_months = (
                    (end_date.year - start_date.year) * 12
                    + end_date.month
                    - start_date.month
                    + 1
                )
                if 1 <= calendar_months <= 120:
                    allowed.add((str(calendar_months), "개월"))

    age = persona.get("age")
    if age is not None:
        allowed.add((str(age), "세"))
        allowed.add((str(age), "살"))
    for boundary in (policy.get("age_min"), policy.get("age_max")):
        if boundary is not None:
            allowed.add((str(boundary), "세"))
            allowed.add((str(boundary), "살"))
    return allowed


def _source_dates(persona: dict, policy: dict) -> tuple[set[tuple[int, int, int]], set[tuple[int, int]]]:
    source_text = json.dumps(
        {"persona": persona, "policy": policy},
        ensure_ascii=False,
        default=str,
    )
    full_dates = {
        (int(year), int(month), int(day))
        for pattern in (_SOURCE_DATE_PATTERN, _KOREAN_FULL_DATE_PATTERN)
        for year, month, day in pattern.findall(source_text)
    }
    month_days = {(month, day) for _, month, day in full_dates}
    month_days.update(
        (int(month), int(day))
        for month, day in _KOREAN_MONTH_DAY_PATTERN.findall(source_text)
    )
    return full_dates, month_days


def _validate_date_facts(
    sentence: str,
    path: str,
    source_dates: set[tuple[int, int, int]],
    source_month_days: set[tuple[int, int]],
) -> tuple[list[str], list[tuple[int, int]]]:
    errors: list[str] = []
    spans: list[tuple[int, int]] = []

    for match in _KOREAN_FULL_DATE_PATTERN.finditer(sentence):
        value = tuple(int(part) for part in match.groups())
        spans.append(match.span())
        if value not in source_dates:
            errors.append(
                f"UNSUPPORTED_DATE_FACT:{path}:{value[0]:04d}-{value[1]:02d}-{value[2]:02d}"
            )

    for match in _KOREAN_MONTH_DAY_PATTERN.finditer(sentence):
        if any(start <= match.start() and match.end() <= end for start, end in spans):
            continue
        value = tuple(int(part) for part in match.groups())
        spans.append(match.span())
        if value not in source_month_days:
            errors.append(
                f"UNSUPPORTED_DATE_FACT:{path}:{value[0]:02d}-{value[1]:02d}"
            )

    return errors, spans


def _validate_summary(result: dict, persona: dict) -> list[str]:
    errors = []
    summary = result.get("persona_summary")
    if not isinstance(summary, dict):
        return errors

    expected_fields = {
        "직업": persona.get("occupation"),
        "성별": persona.get("sex"),
    }
    for field, expected in expected_fields.items():
        if _text(expected) and _normalize(summary.get(field)) != _normalize(expected):
            errors.append(f"PERSONA_SUMMARY_MISMATCH:{field}")

    expected_residence = persona.get("district") or persona.get("province")
    actual_residence = summary.get("거주지")
    if _text(expected_residence):
        expected_region = _normalize(expected_residence)
        actual_region = _normalize(actual_residence)
        if not actual_region or (
            expected_region not in actual_region and actual_region not in expected_region
        ):
            errors.append("PERSONA_SUMMARY_MISMATCH:거주지")

    name = _text(summary.get("이름"))
    expected_name = persona_source_name(persona)
    if expected_name and _normalize(name) != _normalize(expected_name):
        errors.append("PERSONA_SUMMARY_MISMATCH:이름")
    if any(marker in name for marker in ("<", ">", "새로 지은", "미정", "없음")):
        errors.append("PERSONA_SUMMARY_PLACEHOLDER:이름")
    return errors


def _validate_grounding_contract(
    result: dict,
    persona: dict,
    policy: dict,
) -> list[str]:
    grounding = result.get("grounding")
    if not isinstance(grounding, dict):
        return ["GROUNDING_CONTRACT_MISSING"]

    errors = []
    expected = build_grounding_facts(persona, policy)
    for field, expected_value in expected.items():
        if _text(grounding.get(field)) != expected_value:
            errors.append(f"GROUNDING_MISMATCH:{field}")
    return errors


def _validate_complaint_bases(result: dict) -> list[str]:
    errors = []
    complaints = result.get("complaints")
    if not isinstance(complaints, list):
        return errors

    normalized_pairs = []
    for index, complaint in enumerate(complaints):
        if not isinstance(complaint, dict):
            continue
        basis = _text(complaint.get("basis"))
        if basis not in ALLOWED_COMPLAINT_BASES:
            normalized_basis = _normalize(basis)[:40] or "empty"
            errors.append(
                f"COMPLAINT_BASIS_INVALID:complaints[{index}].basis:{normalized_basis}"
            )
        pair = (
            _normalize(complaint.get("complaint_text")),
            _normalize(complaint.get("dialogue")),
        )
        if pair in normalized_pairs:
            errors.append(f"DUPLICATE_COMPLAINT:complaints[{index}]")
        normalized_pairs.append(pair)
    return errors


def _validate_document_procedure_claims(result: dict, policy: dict) -> list[str]:
    """구비서류 이름과 별개인 발급·제출 채널을 지어내지 못하게 한다."""

    complaints = result.get("complaints")
    if not isinstance(complaints, list):
        return []
    required_documents = _text(_policy_detail(policy).get("구비서류"))
    application_method = _text(_policy_detail(policy).get("신청방법"))
    known_application_channels = _known_application_channels(application_method)
    known_channels = set(
        _DOCUMENT_PROCEDURE_CHANNEL_PATTERN.findall(required_documents)
    )
    errors = []
    for index, complaint in enumerate(complaints):
        if (
            not isinstance(complaint, dict)
            or _text(complaint.get("basis")) != "구비서류"
        ):
            continue
        entries = [
            (f"complaints[{index}].{field}", _text(complaint.get(field)))
            for field in ("complaint_text", "dialogue")
        ]
        for path, sentence in _sentences(entries):
            for channel in known_application_channels:
                if _KNOWN_CHANNEL_AVAILABILITY_QUESTION_PATTERNS[channel].search(
                    sentence
                ):
                    errors.append(f"CONTRADICTED_POLICY_FACT:신청방법:{path}")
            if _UNCERTAINTY_PATTERN.search(
                sentence
            ) or _DOCUMENT_PROCEDURE_UNCERTAINTY_PATTERN.search(sentence):
                continue
            sentence_channels = set(
                _DOCUMENT_PROCEDURE_CHANNEL_PATTERN.findall(sentence)
            )
            if not sentence_channels.difference(known_channels):
                continue
            if not _DOCUMENT_PROCESS_ANCHOR_PATTERN.search(sentence):
                continue
            if _DOCUMENT_PROCEDURE_ASSERTION_PATTERN.search(sentence):
                errors.append(f"UNSUPPORTED_DOCUMENT_PROCEDURE:{path}")
    return errors


def _validate_document_identity(result: dict, policy: dict) -> list[str]:
    complaints = result.get("complaints")
    if not isinstance(complaints, list):
        return []
    required_documents = _text(_policy_detail(policy).get("구비서류"))
    alias_groups = required_document_alias_groups(required_documents)
    if not alias_groups:
        return []
    errors = []
    for index, complaint in enumerate(complaints):
        if (
            not isinstance(complaint, dict)
            or _text(complaint.get("basis")) != "구비서류"
        ):
            continue
        combined = _normalize(
            " ".join(
                _text(complaint.get(field))
                for field in ("complaint_text", "dialogue")
            )
        )
        if any(
            not any(_normalize(alias) in combined for alias in aliases)
            for aliases in alias_groups
        ):
            errors.append(f"DOCUMENT_IDENTITY_MISSING:complaints[{index}]")
    return errors


def _directly_causes_exclusion(
    sentence: str,
    cause_patterns: tuple[re.Pattern[str], ...],
) -> bool:
    outcomes = list(_EXCLUSION_PATTERN.finditer(sentence))
    for pattern in cause_patterns:
        for cause in pattern.finditer(sentence):
            if _CAUSE_NEGATION_SUFFIX_PATTERN.search(
                sentence[cause.end() : cause.end() + 24]
            ):
                continue
            for outcome in outcomes:
                if outcome.start() >= cause.end():
                    bridge = sentence[cause.end() : outcome.start()]
                    if len(bridge) > 48:
                        continue
                    if _CAUSAL_BRIDGE_BREAK_PATTERN.search(
                        bridge
                    ) or _ALTERNATE_CAUSE_PATTERN.search(bridge):
                        continue
                    return True
                if outcome.end() <= cause.start():
                    bridge = sentence[outcome.end() : cause.start()]
                    if _REVERSE_CAUSE_BRIDGE_PATTERN.search(bridge):
                        return True
    return False


def _validate_structured_eligibility(
    entries: list[tuple[str, str]],
    persona: dict,
    policy: dict,
) -> list[str]:
    errors = []
    facts = build_grounding_facts(persona, policy)
    age = _text(persona.get("age"))
    province = _text(persona.get("province"))

    for path, sentence in _sentences(entries):
        if _EXCLUSION_NEGATION_PATTERN.search(sentence):
            continue
        if not _EXCLUSION_PATTERN.search(sentence):
            continue

        if facts["age_status"] == MATCHED:
            age_cause_patterns = [
                _AGE_MISMATCH_PATTERN,
                _AGE_DIRECT_CAUSE_PATTERN,
            ]
            if age:
                age_cause_patterns.append(
                    re.compile(
                        rf"(?<!\d){re.escape(age)}\s*(?:세|살)\s*"
                        r"(?:가\s*되면|라(?:서)?|여서|이어서|때문에|"
                        r"라는\s*사실만으로)"
                    )
                )
            if _directly_causes_exclusion(
                sentence,
                tuple(age_cause_patterns),
            ):
                errors.append(f"AGE_ELIGIBILITY_CONTRADICTION:{path}")

        if facts["region_status"] == MATCHED:
            region_cause_patterns = [
                _REGION_MISMATCH_PATTERN,
                _REGION_DIRECT_CAUSE_PATTERN,
            ]
            if province:
                escaped_province = re.escape(province)
                province_identity = re.compile(
                    escaped_province
                    + r"\s*(?:시민|도민|주민|거주자|사람)\s*(?:이)?라(?:서)?"
                )
                province_residence = re.compile(
                    escaped_province
                    + r"(?:에|에서)?\s*(?:살|거주)\S{0,8}\s*"
                    r"(?:때문|라서|여서|이어서)"
                )
                province_mismatch = re.compile(
                    escaped_province
                    + r"\s*(?:시민|도민|주민|거주자|사람|관할|소속)"
                    r"(?:이|은|가|는)?\s*(?:아니|아닙|되지 않)"
                )
                province_condition_mismatch = re.compile(
                    escaped_province
                    + r"\s*(?:(?:거주|지역|관할)\s*)?"
                    r"(?:조건|기준|요건|제한)"
                    r"(?:은|는|이|가|을|를|에|에서|상)?\s*"
                    r"(?:불충족|충족하지 못|맞지 않|해당하지 않|"
                    r"벗어나(?!지(?:는)?\s*않)|제한에\s*걸려)"
                )
                province_residence_mismatch = re.compile(
                    escaped_province
                    + r"(?:에|에서)?\s*(?:살지\s*않|거주하지\s*않)"
                )
                province_address_mismatch = re.compile(
                    r"(?:거주지|주소지)(?:가|는|이|은)?\s*"
                    + escaped_province
                    + r"(?:이|가)?\s*(?:아닌|아니|아닙)"
                )
                region_cause_patterns.extend(
                    (
                        province_identity,
                        province_residence,
                        province_mismatch,
                        province_condition_mismatch,
                        province_residence_mismatch,
                        province_address_mismatch,
                    )
                )
            if _directly_causes_exclusion(
                sentence,
                tuple(region_cause_patterns),
            ):
                errors.append(f"REGION_ELIGIBILITY_CONTRADICTION:{path}")
    return errors


def _validate_policy_facts(
    entries: list[tuple[str, str]],
    persona: dict,
    policy: dict,
) -> list[str]:
    errors = []
    detail = _policy_detail(policy)
    structured_status = build_grounding_facts(persona, policy)["structured_status"]
    application_method = _text(detail.get("신청방법"))
    application_period = _text(detail.get("신청기한"))
    required_documents = _text(detail.get("구비서류"))
    allowed_numbers = _allowed_number_units(persona, policy)
    maximum_support_units = _maximum_number_units(detail.get("지원내용"))
    source_dates, source_month_days = _source_dates(persona, policy)

    for path, sentence in _sentences(entries):
        uncertain = bool(_UNCERTAINTY_PATTERN.search(sentence))

        for field, anchor in _POLICY_FIELD_ANCHORS.items():
            if not _text(detail.get(field)):
                continue
            if _claims_field_absence(sentence, anchor):
                errors.append(f"CONTRADICTED_POLICY_FACT:{field}:{path}")

        if _text(detail.get("지원내용")) and _SUPPORT_AMOUNT_UNKNOWN_PATTERN.search(
            sentence
        ):
            errors.append(f"CONTRADICTED_POLICY_FACT:지원내용:{path}")

        if required_documents and _GENERIC_DOCUMENT_UNKNOWN_PATTERN.search(sentence):
            errors.append(f"CONTRADICTED_POLICY_FACT:구비서류:{path}")

        if (
            _CURRENT_DENIAL_PATTERN.search(sentence)
            and structured_status != NOT_MATCHED
        ):
            errors.append(f"UNSUPPORTED_CURRENT_OUTCOME:{path}")

        if _PERSONAL_PREPARATION_STATE_PATTERN.search(sentence):
            errors.append(f"UNSUPPORTED_APPLICATION_HISTORY:{path}")

        if _DEADLINE_CAUSALITY_CONTRADICTION_PATTERN.search(sentence):
            errors.append(f"DEADLINE_CAUSALITY_CONTRADICTION:{path}")

        if not uncertain and _OVERALL_ELIGIBILITY_ASSERTION_PATTERN.search(sentence):
            errors.append(f"UNSUPPORTED_OVERALL_ELIGIBILITY:{path}")

        if not uncertain and _CURRENT_OUTCOME_ASSERTION_PATTERN.search(sentence):
            errors.append(f"UNSUPPORTED_CURRENT_OUTCOME:{path}")

        if not uncertain and _APPLICATION_HISTORY_PATTERN.search(sentence):
            errors.append(f"UNSUPPORTED_APPLICATION_HISTORY:{path}")

        if not application_method and not uncertain:
            if _CHANNEL_PATTERN.search(sentence) and _METHOD_ASSERTION_PATTERN.search(
                sentence
            ):
                errors.append(f"UNSUPPORTED_POLICY_FACT:신청방법:{path}")

        if not application_period and not uncertain:
            deadline_claim = bool(
                re.search(
                    r"(?:신청|접수|모집|기한|마감).{0,24}"
                    r"(?:오늘|내일|선착순|마감됐|기한이 지났)",
                    sentence,
                )
                or re.search(
                    r"(?:오늘|내일|선착순|조기).{0,16}(?:신청|접수|마감)",
                    sentence,
                )
            )
            if deadline_claim:
                errors.append(f"UNSUPPORTED_POLICY_FACT:신청기한:{path}")

        if not required_documents and not uncertain:
            if re.search(
                r"(?:등본|신분증|증명서|계약서|서류\s*\d+개).{0,20}"
                r"(?:제출|필수|필요|준비)",
                sentence,
            ):
                errors.append(f"UNSUPPORTED_POLICY_FACT:구비서류:{path}")

        date_errors, date_spans = _validate_date_facts(
            sentence,
            path,
            source_dates,
            source_month_days,
        )
        errors.extend(date_errors)

        for match in _NUMBER_UNIT_PATTERN.finditer(sentence):
            if any(
                start <= match.start() and match.end() <= end
                for start, end in date_spans
            ):
                continue
            number, unit = match.groups()
            normalized_pair = _canonical_number_unit(number, unit)
            if normalized_pair not in allowed_numbers:
                errors.append(
                    "UNSUPPORTED_NUMERIC_FACT:"
                    f"{path}:{normalized_pair[0]}{normalized_pair[1]}"
                )
            elif normalized_pair in maximum_support_units:
                has_maximum_qualifier = _match_has_maximum_qualifier(sentence, match)
                if not has_maximum_qualifier:
                    errors.append(
                        "POLICY_QUALIFIER_MISSING:지원내용:최대:"
                        f"{path}:{normalized_pair[0]}{normalized_pair[1]}"
                    )
    return errors


_DOCUMENT_ALIAS_PARTICLES = (
    "으로는",
    "에서는",
    "이라는",
    "이라고",
    "입니다",
    "인지",
    "이며",
    "이고",
    "으로",
    "에서",
    "부터",
    "까지",
    "이나",
    "처럼",
    "보다",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "과",
    "와",
    "도",
    "만",
    "의",
    "로",
    "에",
)
_PHONE_CONTACT_PATTERN = re.compile(
    r"(?<!\d)(?:\+?82[-.\s]?)?"
    r"(?:0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}|"
    r"0\d{1,2}[-.\s]?\d{2,4}|1\d{2,3}[-.\s]?\d{3,4})(?!\d)"
)
_BARE_HOTLINE_PATTERN = re.compile(r"(?<!\d)1\d{2,3}(?!\d)")
_EMAIL_CONTACT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.+-])[A-Za-z0-9_.+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_URL_CONTACT_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+",
    re.IGNORECASE,
)
_AGENCY_NAME_PATTERN = re.compile(
    r"(?<![가-힣A-Za-z0-9·])"
    r"([가-힣A-Za-z0-9·]{2,20}"
    r"(?:\s+[가-힣A-Za-z0-9·]{2,20})?\s*"
    r"(?:행정복지센터|주민센터|복지센터|지원센터|"
    r"지원과|지원팀|복지과|정책과|행정과|주거과|"
    r"사업부|복지부|행정부|센터|공단|공사|위원회|재단|"
    r"복지관|구청|시청|도청|군청|기관|부서))"
)
_GENERIC_AGENCY_NAMES = {
    "담당기관",
    "해당기관",
    "공식기관",
    "관계기관",
    "관련기관",
    "소관기관",
    "운영기관",
    "신청기관",
    "문의기관",
    "담당부서",
    "해당부서",
    "관계부서",
    "관련부서",
    "소관부서",
}
_GENERIC_AGENCY_ROLE_PATTERN = re.compile(
    r"(?:(?:해당|본|이|입력된)?(?:사업|정책)(?:의)?)?"
    r"(?:담당|소관|관계|관련|운영|신청|문의)(?:기관|부서)$"
)
_CONTACT_CONTEXT_PATTERN = re.compile(r"문의|연락|상담|담당|전화|콜센터")


def _is_token_character(value: str) -> bool:
    return bool(value) and (value.isalnum() or "가" <= value <= "힣")


def _contains_document_alias(sentence: str, alias: str) -> bool:
    """조사 결합은 허용하되 `신청서류` 같은 더 긴 명사의 부분일치는 막는다."""

    normalized_sentence = unicodedata.normalize("NFKC", sentence)
    alias_parts = re.split(r"\s+", unicodedata.normalize("NFKC", alias).strip())
    alias_pattern = re.compile(
        r"\s*".join(re.escape(part) for part in alias_parts),
        re.IGNORECASE,
    )
    for match in alias_pattern.finditer(normalized_sentence):
        before = normalized_sentence[match.start() - 1 : match.start()]
        if _is_token_character(before):
            continue
        tail = normalized_sentence[match.end() :]
        if not tail or not _is_token_character(tail[0]):
            return True
        for particle in _DOCUMENT_ALIAS_PARTICLES:
            if not tail.startswith(particle):
                continue
            after_particle = tail[len(particle) : len(particle) + 1]
            if not _is_token_character(after_particle):
                return True
    return False


def _phone_contact_values(text: str) -> set[str]:
    values = {
        re.sub(r"\D", "", match.group())
        for match in _PHONE_CONTACT_PATTERN.finditer(text)
    }
    for match in _BARE_HOTLINE_PATTERN.finditer(text):
        window = text[max(0, match.start() - 24) : match.end() + 24]
        if _CONTACT_CONTEXT_PATTERN.search(window):
            values.add(match.group())
    return values


def _normalized_urls(text: str) -> set[str]:
    return {
        match.group().rstrip(".,);]}").lower()
        for match in _URL_CONTACT_PATTERN.finditer(text)
    }


def _validate_grounded_contact_facts(
    entries: list[tuple[str, str]],
    policy: dict,
) -> list[str]:
    """정책 원문에 없는 전화·URL·이메일·담당기관 생성을 차단한다."""

    policy_source = json.dumps(policy, ensure_ascii=False, default=str)
    policy_phones = _phone_contact_values(policy_source)
    policy_urls = _normalized_urls(policy_source)
    policy_emails = {
        match.group().lower() for match in _EMAIL_CONTACT_PATTERN.finditer(policy_source)
    }
    normalized_policy = _normalize(policy_source)
    application_channels = _known_application_channels(
        _policy_detail(policy).get("신청방법")
    )
    errors: list[str] = []

    for path, sentence in _sentences(entries):
        if _phone_contact_values(sentence).difference(policy_phones):
            errors.append(f"UNSUPPORTED_CONTACT_FACT:PHONE:{path}")
        if _normalized_urls(sentence).difference(policy_urls):
            errors.append(f"UNSUPPORTED_CONTACT_FACT:URL:{path}")
        sentence_emails = {
            match.group().lower()
            for match in _EMAIL_CONTACT_PATTERN.finditer(sentence)
        }
        if sentence_emails.difference(policy_emails):
            errors.append(f"UNSUPPORTED_CONTACT_FACT:EMAIL:{path}")

        for match in _AGENCY_NAME_PATTERN.finditer(sentence):
            agency = _normalize(match.group(1))
            if (
                agency in _GENERIC_AGENCY_NAMES
                or _GENERIC_AGENCY_ROLE_PATTERN.fullmatch(agency)
                or agency in normalized_policy
            ):
                continue
            if (
                "visit" in application_channels
                and ("주민센터" in agency or "행정복지센터" in agency)
            ):
                continue
            errors.append(f"UNSUPPORTED_CONTACT_FACT:AGENCY:{path}")
    return errors


def _validate_grounded_document_and_channel_facts(
    entries: list[tuple[str, str]],
    policy: dict,
) -> list[str]:
    """정책에 없는 구비서류 이름과 신청 채널의 단정적 추가를 검출한다."""

    detail = _policy_detail(policy)
    required_documents = _text(detail.get("구비서류"))
    application_method = _text(detail.get("신청방법"))
    known_documents = set(canonical_document_names(required_documents))
    channel_patterns = {
        **_KNOWN_APPLICATION_CHANNEL_PATTERNS,
        "phone": re.compile(r"전화"),
        "email": re.compile(r"이메일|전자우편"),
        "post": re.compile(r"우편"),
        "fax": re.compile(r"팩스"),
        "government24": re.compile(r"정부24"),
        "district_office": re.compile(r"구청"),
        "city_hall": re.compile(r"시청"),
    }
    known_channels = {
        name
        for name, pattern in channel_patterns.items()
        if pattern.search(application_method)
    }

    errors: list[str] = []
    for path, sentence in _sentences(entries):
        for _, aliases, canonical in _DOCUMENT_CONCEPTS:
            if canonical in known_documents:
                continue
            if any(_contains_document_alias(sentence, alias) for alias in aliases):
                errors.append(
                    f"UNSUPPORTED_POLICY_FACT:구비서류:{path}:{canonical}"
                )

        if not re.search(r"신청|접수", sentence):
            continue
        sentence_channels = {
            name
            for name, pattern in channel_patterns.items()
            if pattern.search(sentence)
        }
        for channel in sorted(sentence_channels.difference(known_channels)):
            errors.append(f"UNSUPPORTED_POLICY_FACT:신청방법:{path}:{channel}")
    return errors


def validate_policy_grounded_text(
    text: str,
    persona: dict,
    policy: dict,
    *,
    path: str = "response",
) -> list[str]:
    """단일 생성 문장의 정책·페르소나 근거 위반을 공개 API로 검사한다."""

    if not isinstance(text, str) or not text.strip():
        return [f"POLICY_GROUNDED_TEXT_EMPTY:{path}"]
    entries = [(path, text.strip())]
    errors = [
        *_validate_structured_eligibility(entries, persona, policy),
        *_validate_policy_facts(entries, persona, policy),
        *_validate_grounded_contact_facts(entries, policy),
        *_validate_grounded_document_and_channel_facts(entries, policy),
        *_validate_persona_facts(entries, persona),
    ]
    return list(dict.fromkeys(errors))


def _validate_persona_facts(
    entries: list[tuple[str, str]],
    persona: dict,
) -> list[str]:
    errors = []
    source = " ".join(
        _text(persona.get(field))
        for field in ("persona", "professional_persona", "family_persona")
    )
    source_sentences = re.split(r"(?<=[.!?。])\s+|\n+", source)
    source_has_housing_status = bool(_PERSONA_HOUSING_SOURCE_PATTERN.search(source))
    source_has_specific_family_expense = bool(
        _SPECIFIC_FAMILY_EXPENSE_ASSERTION_PATTERN.search(source)
    )

    for path, sentence in _sentences(entries):
        if (
            _PERSONAL_HOUSING_ASSERTION_PATTERN.search(sentence)
            and not source_has_housing_status
        ):
            errors.append(f"UNSUPPORTED_PERSONA_FACT:주거:{path}")
        if (
            _SPECIFIC_FAMILY_EXPENSE_ASSERTION_PATTERN.search(sentence)
            and not source_has_specific_family_expense
        ):
            errors.append(f"UNSUPPORTED_FAMILY_FINANCIAL_FACT:{path}")

    for keyword in _CAPABILITY_KEYWORDS:
        source_has_skill = any(
            keyword in sentence and _POSITIVE_SKILL_PATTERN.search(sentence)
            for sentence in source_sentences
        )
        if not source_has_skill:
            continue
        for path, sentence in _sentences(entries):
            if keyword in sentence and _NEGATIVE_SKILL_PATTERN.search(sentence):
                errors.append(f"PERSONA_FACT_CONTRADICTION:역량:{path}:{keyword}")

    if "자가" in source:
        for path, sentence in _sentences(entries):
            if _SELF_RENT_PATTERN.search(sentence):
                errors.append(f"PERSONA_FACT_CONTRADICTION:주거:{path}")
    return errors


def validate_semantic_quality(
    result: dict,
    persona: dict,
    policy: dict,
) -> list[str]:
    """결정적으로 확인 가능한 의미 위반만 오류 코드로 반환한다."""

    entries = _generated_entries(result)
    errors = [
        *_validate_summary(result, persona),
        *_validate_grounding_contract(result, persona, policy),
        *_validate_complaint_bases(result),
        *_validate_document_identity(result, policy),
        *_validate_document_procedure_claims(result, policy),
        *_validate_structured_eligibility(entries, persona, policy),
        *_validate_policy_facts(entries, persona, policy),
        *_validate_persona_facts(entries, persona),
    ]
    return list(dict.fromkeys(errors))
