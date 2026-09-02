import json
import re
import unicodedata

from backend.ai_simulation_core.llm.llm_gateway import (
    run_llm,
)
from backend.ai_simulation_core.prompts.citizen_prompt import (
    ComplaintFocus,
    _complaint_focus,
    citizen_prompt,
)
from backend.ai_simulation_core.simulations.citizen_quality import (
    ALLOWED_COMPLAINT_BASES,
    normalize_maximum_support_qualifiers,
    validate_semantic_quality,
)


_COMPLAINT_BASIS_ALIASES = {
    "신청절차": "신청방법",
    "접수방법": "신청방법",
    "신청기간": "신청기한",
    "접수기간": "신청기한",
    "제출서류": "구비서류",
    "필요서류": "구비서류",
    "지원혜택": "지원내용",
    "지원금": "지원내용",
    "연락처": "문의처",
    "문의기관": "문의처",
    "소관기관명": "정보미제공",
    "정보부족": "정보미제공",
    "정보누락": "정보미제공",
    "개인사정": "개인상황",
    "개인여건": "개인상황",
}

_CLAUSE_SEPARATOR_PATTERN = re.compile(
    r"[.!?。！？,，;；\n]+|(?:지만|으나|는데|데도|거나)|"
    r"(?:(?<![다라])고|며|면서)(?=\s+(?!싶))|"
    r"(?:그러나|그런데|반면|대신|그리고|또한|한편)"
)
_INFORMATION_ABSENCE_PATTERN = re.compile(
    r"(?:안내|명시|제공|공개|기재|제시|설명)(?:가|이|도)?\s*"
    r"(?:되지\s*않|안\s*되|없|(?:되어|돼)\s*있지\s*않)|"
    r"(?:정보|내용|기준|조건|설명|안내)(?:이|가|은|는|도)?\s*"
    r"(?:없|누락|부족|불분명|불명확)|"
    r"(?:알|확인할)\s*수\s*없|확인(?:이|하기|할)?\s*(?:어렵|불가)|"
    r"명확하지\s*않"
)
_INFORMATION_AVAILABLE_PATTERN = re.compile(
    r"(?:안내|명시|제공|공개|기재|제시|설명)(?:가|이|도)?\s*"
    r"(?:(?:되어|돼)\s*있(?!지(?:는)?\s*않)|되었|됐|됨)|"
    r"(?:정보|내용|기준|조건|설명|안내)(?:이|가|은|는|도)?\s*"
    r"(?:있(?!지(?:는)?\s*않)|충분(?!하지)|명확(?!하지))"
)
_ABSENCE_RETRACTION_PATTERN = re.compile(
    r"^\s*(?:(?:은|는)?\s*(?:것|건|게)?(?:은|는|이)?\s*"
    r"(?:아니|아닙|아닌)|"
    r"(?:았|었)?다는\s*(?:말|설명|주장)(?:은|이)?\s*"
    r"(?:틀렸|사실과\s*다르)|"
    r"다는.{0,20}사실이\s*(?:아니|아닙|아닌)|"
    r"다고\s*한\s*(?:것|말|설명)(?:은|이)?\s*(?:사실이\s*)?"
    r"(?:아니|아닙|아닌|틀렸)|(?:되|하)?지(?:는)?\s*않)"
)
_FOCUS_ANCHOR_PATTERNS = {
    "eligibility_gap": re.compile(
        r"지원\s*(?:대상|자격)|자격|대상자|대상\s*(?:조건|기준|요건|범위)|"
        r"누가.{0,16}(?:혜택|지원).{0,8}받"
    ),
    "selection_criteria_gap": re.compile(
        r"선정|선발|심사|평가\s*(?:조건|기준|방법|절차)|"
        r"(?:우선\s*)?(?:뽑|고르)"
    ),
    "exclusion_gap": re.compile(
        r"제외|탈락\s*(?:조건|기준|사유)|지원에서\s*(?:빠지|배제)|빠지|배제"
    ),
    "missing_신청방법": re.compile(
        r"신청\s*(?:방법|절차|채널|창구|경로)|"
        r"접수\s*(?:방법|절차|채널|창구|경로|안내)|"
        r"(?:어디서|어떻게).{0,12}(?:신청|접수)|"
        r"어디로.{0,12}(?:신청서|서류).{0,6}(?:내|제출)|"
        r"신청서.{0,8}(?:내|제출)"
    ),
    "missing_신청기한": re.compile(
        r"신청\s*(?:기한|기간|마감)|접수\s*(?:기한|기간|마감)|"
        r"마감\s*(?:일|날짜)|언제.{0,12}(?:신청|접수)|"
        r"(?:신청|접수).{0,8}언제까지"
    ),
    "missing_구비서류": re.compile(
        r"(?:구비|제출|필요)\s*서류|어떤\s*서류|"
        r"증빙\s*(?:서류|자료)?|첨부\s*(?:서류|자료)"
    ),
    "missing_문의처": re.compile(
        r"문의\s*(?:처|기관|방법)|연락처|전화\s*번호|이메일|전자우편|"
        r"담당\s*부서|부서\s*연결|"
        r"어디에.{0,12}문의|"
        r"어느\s*(?:부서|기관|곳).{0,12}(?:전화|문의|연락)|"
        r"전화.{0,8}(?:부서|기관|곳)"
    ),
    "benefit_effectiveness": re.compile(
        r"지원\s*(?:내용|금(?:액)?|액|규모|혜택)|지원(?:이|은)|"
        r"지원하(?:는|기)\s*규모|보조금|수당|금액|효과|실효성"
    ),
}
_OTHER_BASIS_PATTERN = re.compile(
    r"지원\s*(?:내용|금(?:액)?|액|규모|혜택)|보조금|수당|금액|"
    r"지원(?:이|은).{0,8}(?:충분|부족|효과|도움|부담|규모)|"
    r"지원하(?:는|기)\s*규모|혜택|효과|실효성|"
    r"신청\s*(?:방법|절차|기한|기간|채널)|"
    r"접수\s*(?:방법|절차|기한|기간)|"
    r"(?:구비|제출|필요)\s*서류|문의\s*(?:처|기관|방법)|연락처|"
    r"선정\s*(?:기준|방법|절차)|선발\s*(?:기준|방법|절차)|"
    r"심사\s*(?:기준|방법|절차)|제외\s*(?:조건|기준|사유)|"
    r"소득|재산|주택|무주택|중복\s*수급|가구|취업|재직|서류"
)
_AGE_BOUNDARY_RELATION_PATTERN = re.compile(
    r"(?:나이|연령|연령대).{0,16}"
    r"(?:기준|조건|경계|하한|상한|차이|적용|형평|이유)|"
    r"(?:기준|조건|경계|하한|상한|차이|적용|형평|이유).{0,16}"
    r"(?:나이|연령|(?:만\s*)?\d{1,3}\s*(?:세|살))|"
    r"(?:만\s*)?\d{1,3}\s*(?:세|살).{0,20}"
    r"(?:기준|경계|하한|상한|차이|이유|적용|형평|이상|이하|미만|초과)|"
    r"(?:한|1)\s*살\s*(?:차이|낮|높|어리|많)"
)
_REGION_BOUNDARY_RELATION_PATTERN = re.compile(
    r"(?:지역|거주지|주소지|관할|행정구역|시|도|구|군).{0,18}"
    r"(?:기준|조건|경계|불일치|차이|다르|제한|적용|형평|대안|제외)|"
    r"(?:기준|조건|경계|불일치|제한|적용|형평|대안).{0,18}"
    r"(?:지역|거주지|주소지|관할|행정구역|시|도|구|군)|"
    r"(?:정책\s*지역).{0,40}(?:저|제|현재\s*거주지)"
)
_AGE_COMPLAINT_PATTERN = re.compile(
    r"(?:나이|연령|연령대).{0,20}"
    r"(?:궁금|의문|왜|이유|설명|완화|개선|형평|차이|미달|초과|"
    r"낮|높|제외|대상이\s*아니)|"
    r"(?:궁금|의문|왜|이유|설명|완화|개선|형평).{0,20}"
    r"(?:나이|연령|연령대)"
)
_REGION_COMPLAINT_PATTERN = re.compile(
    r"(?:지역|거주지|주소지|관할|행정구역).{0,20}"
    r"(?:궁금|의문|왜|이유|설명|완화|개선|형평|차이|불일치|"
    r"다르|제한|경계|대안|제외)|"
    r"(?:궁금|의문|왜|이유|설명|완화|개선|형평|대안).{0,20}"
    r"(?:지역|거주지|주소지|관할|행정구역)"
)
_AGE_VALUE_COMPLAINT_PATTERN = re.compile(
    r"(?<!\d)\d{1,3}\s*(?:세|살).{0,8}"
    r"(?:나이|연령)?\s*(?:기준|조건|경계|제한).{0,10}"
    r"(?:궁금|의문|설명|완화|개선|형평|확대|낮|높)|"
    r"(?<!\d)\d{1,3}\s*(?:세|살).{0,20}"
    r"(?<!\d)\d{1,3}\s*(?:세|살).{0,12}"
    r"(?:구분|경계|제한).{0,10}(?:없애|완화|개선|확대)"
)
_AGE_GROUP_COMPLAINT_PATTERN = re.compile(
    r"(?:청년층?|중장년층?|장년층?|노년층?|미성년자?|성인).{0,28}"
    r"(?:만\s*받|못\s*받|제한|완화|구분|없애|제외)|"
    r"(?:스무|서른|마흔|쉰|예순|일흔|여든|아흔)\s*살.{0,16}"
    r"(?:미만|이상|이하|초과).{0,16}(?:못\s*받|제한|완화|제외)"
)
_KOREAN_REGION_TOKEN = (
    r"(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|"
    r"충청북|충청남|충북|충남|전북|전남|경상북|경상남|경북|경남|제주|"
    r"[가-힣]{1,10}(?:특별시|광역시|특별자치시|특별자치도|시|군|구))"
)
_REGION_SCOPE_REQUEST_PATTERN = re.compile(
    rf"{_KOREAN_REGION_TOKEN}\s*(?:보다|에\s*만|에도|도|까지|외에도|이외에도|만|뿐)"
    r".{0,28}(?:지원|적용|포함|확대|허용|대상|완화|넓혀)|"
    rf"{_KOREAN_REGION_TOKEN}\s*(?:시민|도민|주민|거주자).{{0,24}}"
    r"(?:만\s*받|못\s*받|제외|제한|넓혀|확대)|"
    rf"{_KOREAN_REGION_TOKEN}.{{0,10}}{_KOREAN_REGION_TOKEN}.{{0,16}}"
    r"(?:구분|경계|차이|제한).{0,12}(?:없애|완화|개선|확대)"
)
_UNRELATED_FOCUS_PATTERN = re.compile(
    r"정책\s*(?:이름|명칭|제목)|글씨|글자|문구|색상|표시\s*(?:방식|색)|"
    r"디자인|화면|폰트|홈페이지|웹\s*사이트|사이트|로그인|서버\s*오류|"
    r"담당자\s*(?:태도|응대)"
)
_AGE_BOUNDARY_DENIAL_PATTERN = re.compile(
    r"(?:낮|높|미달|초과|벗어나)(?:지(?:는)?\s*않|은\s*(?:것|건).{0,4}"
    r"(?:아니|아닙))|(?:한\s*살\s*)?차이(?:가|는)?\s*없|"
    r"(?:나이|연령|한\s*살\s*차이|연령\s*경계).{0,16}"
    r"문제(?:가|는|도)?\s*(?:아니|아닙|없)"
)
_REGION_BOUNDARY_DENIAL_PATTERN = re.compile(
    r"(?:다르|불일치|벗어나|맞지\s*않)(?:지(?:는)?\s*않|은\s*(?:것|건).{0,4}"
    r"(?:아니|아닙))|(?:지역\s*)?차이(?:가|는)?\s*없|"
    r"(?:지역|거주지|주소지|지역\s*경계|지역\s*차이).{0,16}"
    r"문제(?:가|는|도)?\s*(?:아니|아닙|없)"
)
_COMPLAINT_CONCERN_PATTERN = re.compile(
    r"궁금|의문|걱정|부족|충분하지|어렵|불편|부담|촉박|짧|복잡|"
    r"개선|완화|늘려|확대|확인할\s*수\s*없|안내(?:해|가\s*필요)|"
    r"설명(?:해|이\s*필요)|알려\s*주|(?:가능|충분|적절)한지"
)
_NO_COMPLAINT_PATTERN = re.compile(
    r"아무(?:런)?\s*문제.{0,5}없|문제(?:가|는)?\s*없|"
    r"전혀.{0,12}(?:부담|어렵|불편|문제).{0,6}(?:없|않)|"
    r"(?:충분|쉽|편리|좋).{0,12}(?:문제|부담).{0,6}(?:없|않)"
)
_RESOLVED_CONCERN_PATTERN = re.compile(
    r"(?:충분한지|편리한지|가능한지|적절한지).{0,24}"
    r"(?:분명|명확|확실)(?!하지\s*않)|"
    r"(?:충분한지|편리한지|가능한지|적절한지).{0,24}"
    r"확인(?:했|됐).{0,16}(?:충분|편리|가능|적절|쉽|문제\s*없)|"
    r"(?:충분한지|편리한지|가능한지|적절한지).{0,24}"
    r"(?:살펴(?:봤|보았|보면)).{0,16}(?:충분|편리|가능|적절|쉽)|"
    r"(?:어렵|불편|복잡|촉박|짧)(?:하?지(?:는)?\s*않|하?지\s*않다는)|"
    r"(?:문제|부담|걱정)(?:이|가|은|는)?\s*없"
)
_NON_TARGET_GAP_CONTENT_PATTERN = re.compile(
    r"(?:자)?(?:의|에\s*대한)\s*(?:제\s*)?(?:의견|요청|생각|경험|상황)|"
    r"(?:자)?에게|"
    r"(?:항목|란|페이지|문서|화면)\s*(?:에|에는|에서)"
)
_POLICY_REGION_ROLE_PATTERN = (
    r"정책(?:의|이|가|은|는)?(?:\s*(?:적용\s*)?지역)?|"
    r"적용\s*지역|정책\s*대상\s*지역"
)
_PERSONA_REGION_ROLE_PATTERN = (
    r"(?:현재\s*)?(?:제\s*)?(?:거주지|주소지)|저는|제가|본인은|"
    r"현재\s*(?:살고\s*있는|사는)\s*곳"
)
_SAFE_FOCUS_PATTERNS = {
    "benefit_effectiveness": (
        _FOCUS_ANCHOR_PATTERNS["benefit_effectiveness"],
        re.compile(
            r"충분한(?:지|가)|충분하지|부족|효과.{0,10}(?:있는지|궁금|의문|설명)|"
            r"실효.{0,10}(?:궁금|의문|설명)|부담.{0,12}(?:줄|덜|완화)|"
            r"적정한지|늘려|확대|"
            r"(?:지원|혜택|금액|규모).{0,18}(?:걱정|의문|궁금)"
        ),
    ),
    "application_accessibility": (
        re.compile(r"신청\s*(?:방법|절차|채널)|접수\s*(?:방법|절차|채널)"),
        re.compile(
            r"접근.{0,10}(?:어렵|불편|가능한지|편리한지|개선|궁금|걱정)|"
            r"편리한지|이용.{0,10}(?:어렵|불편|가능한지)|"
            r"절차.{0,10}(?:복잡|어렵|불편|안내|설명)|"
            r"(?:신청|접수|접근|절차|이용).{0,18}"
            r"(?:불편|어렵|충분한지|궁금|걱정)"
        ),
    ),
    "deadline_burden": (
        re.compile(r"신청\s*(?:기한|기간|마감)|접수\s*(?:기한|기간|마감)"),
        re.compile(
            r"충분한지|충분하지|촉박|짧|시간.{0,8}(?:부족|충분한지)|"
            r"여유.{0,8}(?:있는지|없)|준비.{0,8}(?:어렵|부담|가능한지)|"
            r"(?:신청|접수|기한|기간|마감|준비|시간).{0,18}"
            r"(?:부담|궁금|걱정)"
        ),
    ),
    "document_burden": (
        re.compile(r"(?:구비|제출|필요)\s*서류|서류\s*(?:항목|목록)"),
        re.compile(
            r"준비.{0,10}(?:어렵|부담|어떻게|안내|궁금)|"
            r"발급.{0,10}(?:어렵|부담|어떻게|안내|궁금)|"
            r"제출.{0,10}(?:어렵|부담|어떻게|안내|궁금)|"
            r"절차.{0,10}(?:안내|설명|어렵|궁금)|"
            r"(?:서류|준비|발급|제출).{0,18}(?:부담|궁금|걱정)"
        ),
    ),
}


def _span_distance(left: re.Match, right: re.Match) -> int:
    if left.end() < right.start():
        return right.start() - left.end()
    if right.end() < left.start():
        return left.start() - right.end()
    return 0


def _region_value_patterns(value: str, province: str = "") -> list[str]:
    value_text = unicodedata.normalize("NFKC", str(value or "")).strip()
    candidates = {value_text}
    prefixed = re.fullmatch(r"\s*([^\s\-_·]+)\s*[\-_·]\s*(.+?)\s*", value_text)
    normalized_province = re.sub(r"[\s\-_·]", "", province)
    if (
        prefixed
        and normalized_province
        and re.sub(r"[\s\-_·]", "", prefixed.group(1)) == normalized_province
    ):
        candidates.add(prefixed.group(2))

    patterns = []
    for candidate in candidates:
        parts = [part for part in re.split(r"[\s\-_·]+", candidate) if part]
        if not parts:
            continue
        body = r"[\s\-_·]*".join(re.escape(part) for part in parts)
        patterns.append(
            rf"(?<![가-힣A-Za-z0-9]){body}"
            rf"(?=$|[\s,.;:!?/()\-_·]|"
            rf"(?:은|는|이|가|을|를|에|에서|와|과|로|으로|인|이고|이며|인데|"
            rf"라서|입니다|만|도|까지))"
        )
    return patterns


def _role_targets_region(
    text: str,
    expected_value: str,
    other_value: str,
    expected_province: str,
    other_province: str,
    role_pattern: str,
) -> bool:
    roles = list(re.finditer(role_pattern, text))
    expected_matches = [
        match
        for pattern in _region_value_patterns(expected_value, expected_province)
        for match in re.finditer(pattern, text)
    ]
    other_matches = [
        match
        for pattern in _region_value_patterns(other_value, other_province)
        for match in re.finditer(pattern, text)
    ]
    for role in roles:
        expected_distance = min(
            (_span_distance(role, match) for match in expected_matches),
            default=None,
        )
        other_distance = min(
            (_span_distance(role, match) for match in other_matches),
            default=None,
        )
        if (
            expected_distance is not None
            and expected_distance <= 16
            and (other_distance is None or expected_distance < other_distance)
        ):
            return True
    return False


def _role_targets_age(
    text: str,
    expected_number: str,
    other_numbers: set[str],
    *,
    policy_role: bool,
) -> bool:
    if policy_role:
        marker = r"정책(?:의|상|이|가|은|는)?|지원\s*대상(?:\s*연령)?"
    else:
        marker = r"현재|저는|제가|제\s*나이|본인은|나는|만"

    roles = list(re.finditer(marker, text))
    expected_matches = list(
        re.finditer(
            rf"(?<!\d){re.escape(expected_number)}\s*(?:세|살)",
            text,
        )
    )
    other_matches = [
        match
        for number in other_numbers
        for match in re.finditer(
            rf"(?<!\d){re.escape(number)}\s*(?:세|살)",
            text,
        )
    ]
    for role in roles:
        expected_distance = min(
            (_span_distance(role, match) for match in expected_matches),
            default=None,
        )
        other_distance = min(
            (_span_distance(role, match) for match in other_matches),
            default=None,
        )
        limit = 12 if policy_role else 10
        if (
            expected_distance is not None
            and expected_distance <= limit
            and (other_distance is None or expected_distance < other_distance)
        ):
            return True

    if policy_role:
        return bool(
            re.search(
                rf"(?<!\d){re.escape(expected_number)}\s*(?:세|살)\s*"
                r"(?:이상|이하|미만|초과|하한|상한)",
                text,
            )
        )
    return bool(
        re.search(
            rf"(?<!\d){re.escape(expected_number)}\s*(?:세|살)\s*"
            r"(?:인|인\s*저는|인\s*제가|인\s*본인은)",
            text,
        )
    )


def _has_explicit_information_gap(text: str, focus_kind: str) -> bool:
    target_pattern = _FOCUS_ANCHOR_PATTERNS[focus_kind]
    competing_patterns = {
        kind: pattern
        for kind, pattern in _FOCUS_ANCHOR_PATTERNS.items()
        if kind != focus_kind
    }
    allowed_embedded = {
        "selection_criteria_gap": {"eligibility_gap"},
        "exclusion_gap": {"eligibility_gap"},
    }.get(focus_kind, set())
    found_target_gap = False
    clauses = [
        clause.strip()
        for clause in _CLAUSE_SEPARATOR_PATTERN.split(text)
        if clause.strip()
    ]
    for clause in clauses:
        targets = list(target_pattern.finditer(clause))
        raw_absences = list(_INFORMATION_ABSENCE_PATTERN.finditer(clause))
        retracted_absences = [
            match
            for match in raw_absences
            if _ABSENCE_RETRACTION_PATTERN.search(clause[match.end() : match.end() + 32])
        ]
        absences = [match for match in raw_absences if match not in retracted_absences]
        competitor_matches = {
            kind: list(pattern.finditer(clause))
            for kind, pattern in competing_patterns.items()
        }
        present_competitors = {
            kind for kind, matches in competitor_matches.items() if matches
        }

        if not targets:
            has_issue_signal = bool(
                raw_absences
                or _COMPLAINT_CONCERN_PATTERN.search(clause)
            )
            if present_competitors and has_issue_signal:
                return False
            continue

        issue_competitors = set()
        for kind, matches in competitor_matches.items():
            for match in matches:
                if any(
                    _span_distance(match, signal) <= 48
                    for signal in (
                        *raw_absences,
                        *_COMPLAINT_CONCERN_PATTERN.finditer(clause),
                    )
                ):
                    issue_competitors.add(kind)
                    break
        if issue_competitors - allowed_embedded:
            return False
        for contradicted in (
            *retracted_absences,
            *_INFORMATION_AVAILABLE_PATTERN.finditer(clause),
        ):
            if any(_span_distance(target, contradicted) <= 48 for target in targets):
                return False

        for absence in absences:
            for target in targets:
                if target.end() > absence.start():
                    continue
                if _span_distance(target, absence) > 48:
                    continue
                bridge = clause[target.end() : absence.start()]
                if _NON_TARGET_GAP_CONTENT_PATTERN.search(bridge):
                    continue
                found_target_gap = True
    return found_target_gap


def _has_competing_safe_focus(
    text: str,
    focus_kind: str,
) -> bool:
    equivalent_anchor = {
        "benefit_effectiveness": "benefit_effectiveness",
        "application_accessibility": "missing_신청방법",
        "deadline_burden": "missing_신청기한",
        "document_burden": "missing_구비서류",
    }[focus_kind]
    for clause in _CLAUSE_SEPARATOR_PATTERN.split(text):
        for other_kind, (topic_pattern, concern_pattern) in _SAFE_FOCUS_PATTERNS.items():
            if other_kind == focus_kind:
                continue
            if topic_pattern.search(clause) and concern_pattern.search(clause):
                return True
        for kind, pattern in _FOCUS_ANCHOR_PATTERNS.items():
            if kind == equivalent_anchor:
                continue
            if pattern.search(clause) and _INFORMATION_ABSENCE_PATTERN.search(clause):
                return True
    return False


def _canonicalize_complaint_basis(value: object) -> object:
    if not isinstance(value, str):
        return value
    compact = re.sub(
        r"[\s\-_·]",
        "",
        unicodedata.normalize("NFKC", value),
    )
    if compact in ALLOWED_COMPLAINT_BASES:
        return compact
    return _COMPLAINT_BASIS_ALIASES.get(compact, value.strip())


def _normalize_complaint_bases(result: dict) -> dict:
    complaints = result.get("complaints")
    if not isinstance(complaints, list):
        return result
    for complaint in complaints:
        if isinstance(complaint, dict) and "basis" in complaint:
            complaint["basis"] = _canonicalize_complaint_basis(complaint["basis"])
    return result


def _focus_alignment_errors(
    complaint: dict,
    focus: ComplaintFocus,
    persona: dict,
    policy: dict,
) -> list[str]:
    complaint_fields = {
        field: str(complaint.get(field) or "")
        for field in ("complaint_text", "dialogue")
    }
    if focus.kind == "age_boundary":
        policy_numbers = set(re.findall(r"\d+", focus.policy_evidence))
        persona_numbers = set(re.findall(r"\d+", focus.persona_evidence))
        expected_numbers = policy_numbers | persona_numbers
        fields_are_aligned = all(
            all(
                re.search(rf"(?<!\d){re.escape(number)}\s*(?:세|살)", text)
                for number in expected_numbers
            )
            and all(
                _role_targets_age(
                    text,
                    number,
                    persona_numbers,
                    policy_role=True,
                )
                for number in policy_numbers
            )
            and all(
                _role_targets_age(
                    text,
                    number,
                    policy_numbers,
                    policy_role=False,
                )
                for number in persona_numbers
            )
            and _AGE_BOUNDARY_RELATION_PATTERN.search(text)
            and not _OTHER_BASIS_PATTERN.search(text)
            and not _REGION_COMPLAINT_PATTERN.search(text)
            and not _REGION_SCOPE_REQUEST_PATTERN.search(text)
            and not _UNRELATED_FOCUS_PATTERN.search(text)
            and not _AGE_BOUNDARY_DENIAL_PATTERN.search(text)
            for text in complaint_fields.values()
        )
        if not fields_are_aligned:
            return ["COMPLAINT_FOCUS_MISMATCH:age_boundary:missing_age_evidence"]
    elif focus.kind == "region_mismatch":
        policy_region = str(policy.get("region_province") or "").strip()
        persona_region = str(persona.get("province") or "").strip()
        policy_district = str(policy.get("region_district") or "").strip()
        persona_district = str(persona.get("district") or "").strip()

        def field_is_aligned(text: str) -> bool:
            if policy_region != persona_region:
                return _role_targets_region(
                    text,
                    policy_region,
                    persona_region,
                    "",
                    "",
                    _POLICY_REGION_ROLE_PATTERN,
                ) and _role_targets_region(
                    text,
                    persona_region,
                    policy_region,
                    "",
                    "",
                    _PERSONA_REGION_ROLE_PATTERN,
                )
            if policy_district and policy_district != persona_district:
                return _role_targets_region(
                    text,
                    policy_district,
                    persona_district,
                    policy_region,
                    persona_region,
                    _POLICY_REGION_ROLE_PATTERN,
                ) and _role_targets_region(
                    text,
                    persona_district,
                    policy_district,
                    persona_region,
                    policy_region,
                    _PERSONA_REGION_ROLE_PATTERN,
                )
            return False

        if not all(
            field_is_aligned(text)
            and _REGION_BOUNDARY_RELATION_PATTERN.search(text)
            and not _OTHER_BASIS_PATTERN.search(text)
            and not _AGE_COMPLAINT_PATTERN.search(text)
            and not _AGE_VALUE_COMPLAINT_PATTERN.search(text)
            and not _AGE_GROUP_COMPLAINT_PATTERN.search(text)
            and not _UNRELATED_FOCUS_PATTERN.search(text)
            and not _REGION_BOUNDARY_DENIAL_PATTERN.search(text)
            for text in complaint_fields.values()
        ):
            return ["COMPLAINT_FOCUS_MISMATCH:region_mismatch:missing_region_evidence"]
    elif focus.kind in {
        "eligibility_gap",
        "selection_criteria_gap",
        "exclusion_gap",
        "missing_신청방법",
        "missing_신청기한",
        "missing_구비서류",
        "missing_문의처",
    }:
        if not all(
            _has_explicit_information_gap(text, focus.kind)
            and not _UNRELATED_FOCUS_PATTERN.search(text)
            for text in complaint_fields.values()
        ):
            return [f"COMPLAINT_FOCUS_MISMATCH:{focus.kind}:missing_gap_evidence"]
    elif focus.kind in _SAFE_FOCUS_PATTERNS:
        topic_pattern, concern_pattern = _SAFE_FOCUS_PATTERNS[focus.kind]
        if not all(
            topic_pattern.search(text)
            and concern_pattern.search(text)
            and not _NO_COMPLAINT_PATTERN.search(text)
            and not _RESOLVED_CONCERN_PATTERN.search(text)
            and not _UNRELATED_FOCUS_PATTERN.search(text)
            and not _has_competing_safe_focus(
                text,
                focus.kind,
            )
            for text in complaint_fields.values()
        ):
            return [f"COMPLAINT_FOCUS_MISMATCH:{focus.kind}:missing_topic_evidence"]
    else:
        return [f"COMPLAINT_FOCUS_MISMATCH:{focus.kind}:unsupported_focus_kind"]
    return []


def parse_citizen_response(raw_output: str) -> dict | None:
    # 로컬 모델이 JSON 코드 블록을 붙인 경우 마크다운 기호를 제거
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        print(f"[JSON 파싱 실패] {error}\n원본: {raw_output[:200]}")
        return None

    if not isinstance(parsed, dict):
        print("[JSON 구조 오류] 최상위 값은 객체여야 합니다.")
        return None
    return _normalize_complaint_bases(parsed)


def validate_citizen_response(
    result: dict,
    persona: dict,
    policy: dict | None = None,
    expected_focus: ComplaintFocus | None = None,
) -> list[str]:
    """
    검증 실패 사유 목록을 반환. 빈 리스트면 통과.
    """
    errors = []

    if result is None:
        return ["파싱 실패 (None)"]

    persona_summary = result.get("persona_summary", {})
    if not isinstance(persona_summary, dict):
        errors.append("persona_summary가 JSON 객체가 아님")
        persona_summary = {}

    # 1) 나이 필수값과 불일치 체크
    summary_age_raw = persona_summary.get("나이", "")
    summary_age_match = re.fullmatch(
        r"\s*(\d{1,3})(?:\s*세)?\s*",
        str(summary_age_raw),
    )
    summary_age = summary_age_match.group(1) if summary_age_match else ""
    actual_age_raw = persona.get("age")
    actual_age = "" if actual_age_raw is None else str(actual_age_raw)
    if actual_age and not summary_age:
        errors.append("나이 필드가 비어있음")
    elif summary_age and actual_age and summary_age != actual_age:
        errors.append(f"나이 불일치: persona_summary={summary_age}, 실제={actual_age}")

    # 2) 이름 누락 체크
    name = persona_summary.get("이름", "")
    if not isinstance(name, str) or not name.strip():
        errors.append("이름 필드가 비어있거나 문자열이 아님")

    # 3) 성격 및 민원 필수 구조 체크
    personality = result.get("personality", "")
    if not isinstance(personality, str) or not personality.strip():
        errors.append("personality가 비어있거나 문자열이 아님")

    complaints = result.get("complaints", [])
    if not isinstance(complaints, list):
        errors.append("complaints가 JSON 배열이 아님")
        complaints = []
    elif len(complaints) != 1:
        errors.append("complaints는 정확히 1개여야 함")

    valid_complaints = []
    for index, complaint in enumerate(complaints, start=1):
        if not isinstance(complaint, dict):
            errors.append("complaints 항목이 JSON 객체가 아님")
            continue
        valid_complaints.append(complaint)
        for field in ("complaint_text", "dialogue"):
            value = complaint.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"complaints[{index}].{field}가 비어있거나 문자열이 아님")

    locked_basis = expected_focus.basis if expected_focus else None
    if locked_basis and len(valid_complaints) == 1:
        actual_basis = valid_complaints[0].get("basis")
        if actual_basis != locked_basis:
            errors.append(
                f"COMPLAINT_BASIS_MISMATCH:expected={locked_basis}:actual={actual_basis}"
            )
        elif policy is not None:
            errors.extend(
                _focus_alignment_errors(
                    valid_complaints[0],
                    expected_focus,
                    persona,
                    policy,
                )
            )

    # 4) JSON 영문 키를 제외하고 모델이 생성한 실제 문장만 영어 혼용 체크
    generated_texts = [
        str(name),
        str(personality),
        *(str(complaint.get("complaint_text", "")) for complaint in valid_complaints),
        *(str(complaint.get("dialogue", "")) for complaint in valid_complaints),
    ]
    full_text = " ".join(generated_texts)
    if re.search(r"[a-zA-Z]", full_text):
        errors.append("영어 단어 혼용 감지")

    if policy is not None:
        errors.extend(validate_semantic_quality(result, persona, policy))

    return list(dict.fromkeys(errors))


def run_citizen_simulation(
    persona: dict, policy: dict, max_retries: int = 3
) -> dict | None:
    parsed = None
    errors = []
    validation_feedback = None
    focus = _complaint_focus(persona, policy)

    for attempt in range(max_retries):
        prompt = citizen_prompt(
            persona,
            policy,
            validation_feedback=validation_feedback,
            focus=focus,
        )
        # main과 같은 로컬 Qwen 모델을 사용하여 시민 응답 생성
        raw_output = run_llm(prompt)
        parsed = parse_citizen_response(raw_output)

        if parsed is None:
            print(f"  [시도 {attempt + 1}/{max_retries}] JSON 파싱 실패, 재시도")
            errors = ["JSON_PARSE_ERROR"]
            validation_feedback = errors
            continue

        parsed = normalize_maximum_support_qualifiers(parsed, policy)
        errors = validate_citizen_response(
            parsed,
            persona,
            policy,
            expected_focus=focus,
        )
        if not errors:
            if attempt > 0:
                print(f"  [시도 {attempt + 1}/{max_retries}] 검증 통과")
            break
        print(f"  [시도 {attempt + 1}/{max_retries}] 검증 실패: {errors}")
        validation_feedback = errors

    if parsed is None or errors:
        if errors:
            print(f"  [최종 검증 실패] 유효한 시민 응답을 생성하지 못했습니다: {errors}")
        return None

    # 검증을 통과한 결과에만 후속 연결용 페르소나 식별자를 추가한다.
    parsed["persona_id"] = persona.get("uuid")
    parsed["_validation_errors"] = []
    parsed["_quality_gate"] = {
        "version": "citizen-grounding-v1",
        "status": "passed",
        "removed_complaints": 0,
        "generation_attempts": attempt + 1,
    }
    return parsed
