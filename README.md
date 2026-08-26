# CivicEcho

CivicEcho는 정책 시행 전에 지역·나이 조건에 맞는 시민 페르소나 3명을 선택하고,
예상 민원과 정책 사각지대를 검토하는 웹 애플리케이션입니다.

제품 흐름은 다음과 같습니다.

1. 정책을 직접 입력하거나 파일을 업로드합니다.
2. 정책의 지역·나이 조건을 구조화합니다.
3. 기본적으로 조건을 충족하는 시민 페르소나 중 3명을 선택합니다.
   연령 경계 분석이 필요하면 같은 지역의 정책 나이 경계 1년 밖 페르소나를
   명시적으로 직접 선택할 수 있습니다.
4. 로컬 LLM이 시민 민원을 생성하고, 서버가 정책 원문에만 근거한 공무원 답변을 구성합니다.
5. 시민 생성 결과와 공무원 답변이 근거·정합성 검증을 통과한 경우에만 표시합니다.
6. 정부24 유사 정책과 공개 민원 FAQ 참고 사례를 함께 확인합니다.

## 제품 범위

- 정책 입력 → 페르소나 선택 → 예상 민원·사각지대 확인이 현재 제품의 전 범위입니다.
- 법령 분석, 법령 추천, 법령 여론 예측 기능은 포함하지 않습니다.
- 결과는 실제 시민 모집단의 통계적 예측이나 수급 자격 확정이 아닙니다.
- 지역·나이 적합은 페르소나 후보군 선택 기준이며, 소득·재산·서류 등 전체 자격을 확정하지 않습니다.

## 주요 구성

- 백엔드: FastAPI, Qwen3-8B (BF16 원본 가중치, GPU/CPU 자동 분산), ChromaDB
- 프런트엔드: React 19, Vite 8, Tailwind CSS 4
- 페르소나: `nvidia/Nemotron-Personas-Korea` Parquet
- 유사 정책: 정부24 대한민국 공공서비스(혜택) 정보 스냅샷
- 참고 민원: 공개 민원 FAQ 정본과 활성 Chroma 인덱스

## 실행 요구사항

- Python 3.12 이상
- Node.js `^20.19.0` 또는 `>=22.12.0`
- 최신 Chromium 기반 Chrome
- 로컬 Qwen 실행을 위한 CUDA GPU 권장
- 최초 페르소나 다운로드를 위한 네트워크 연결과 약 2GB의 여유 공간

CPU에서도 실행할 수 있지만 실제 시뮬레이션은 매우 느릴 수 있습니다.

## 설치

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

cd frontend
npm ci
cd ..
```

페르소나 Parquet은 Git에 포함하지 않습니다. 첫 페르소나 후보 요청 시
`data/raw/personas/`로 자동 다운로드됩니다. Qwen 생성 모델과 민원 임베딩 모델도
최초 실행 시 Hugging Face 캐시 또는 네트워크 연결이 필요합니다.

## 개발 서버 실행

터미널 1에서 백엔드를 실행합니다.

```bash
.venv/bin/python -m uvicorn backend.api:app \
  --host 127.0.0.1 \
  --port 8000
```

터미널 2에서 프런트엔드를 실행합니다.

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Chrome에서 `http://127.0.0.1:5173`을 열고 다음 흐름을 확인합니다.

```text
/policy → 정책 입력 또는 파일 업로드 → 지역·나이 지정
→ 페르소나 3명 선택 → 실행 → 완료 결과
```

상태 확인:

```bash
curl -f http://127.0.0.1:8000/api/health
curl -f http://127.0.0.1:5173/api/health
```

## 지원 정책 파일

| 형식 | 확장자 |
|---|---|
| PDF | `.pdf` |
| Word | `.docx` |
| 한글 XML 문서 | `.hwpx` |
| 텍스트 | `.txt` |
| Markdown | `.md` |

- 파일당 최대 크기는 15MB입니다.
- 구형 바이너리 `.hwp`는 지원하지 않습니다.
- 업로드 단계는 텍스트에서 정책 필드만 추출하며 활성 정책을 변경하지 않습니다.
- 사용자가 추출값을 검토하고 지역·나이 조건과 페르소나를 확정한 뒤에만 시뮬레이션을 제출합니다.

## 생성 결과 검증과 실패 처리

시민 결과는 구조, 한국어 문장, 정책 근거, 지역·나이 판정, 페르소나 사실,
숫자·신청 채널의 근거, 중복 민원을 검사합니다.

공무원 답변은 자유 생성하지 않습니다. 시민 민원의 단일 쟁점과 입력된 정책 필드를
연결해 결정적으로 구성하며, 입력에 없는 자격·승인·기한·서류·문의처는 확정하지
않습니다. 시민과 공무원 페르소나 ID, 답변 근거, 최종 문장을 API 저장 직전에 다시
검증하고 하나라도 다르면 작업 전체를 실패 처리합니다.

- 민원 하나라도 정책·페르소나 근거 검증에 실패하면 해당 시민 응답 전체를 최대 3회 다시 생성합니다.
- 현재 생성 계약은 시민 1명마다 미리 정한 단일 쟁점의 민원 정확히 1개입니다.
- 검증에 실패한 민원만 제거해 축약 결과를 통과시키는 경로는 없습니다.
- 페르소나 한 명이라도 최종 생성에 실패하면 작업 전체를 `failed`로 처리합니다.
- 실패 작업에는 부분 결과를 노출하지 않습니다.
- 특정한 ‘원하는 민원 유형’이 나오지 않았다는 이유만으로 제품이 결과를 임의 재생성하지는 않습니다. 이 항목은 별도 품질평가에서 coverage 실패로 기록합니다.

## API

| 메서드 | 경로 | 용도 |
|---|---|---|
| GET | `/api/health` | 서버 상태 |
| GET | `/api/personas/options` | 지역·나이 선택지 |
| GET | `/api/personas/candidates` | 조건에 맞는 페르소나 후보 |
| GET | `/api/policies/active` | 현재 활성 정책 |
| POST | `/api/policies/extract-file` | 정책 파일 필드 추출 |
| POST | `/api/policies/similar` | 유사 정책 검색 |
| POST | `/api/policies/direct` | 이전 호출자용 호환 경로 |
| POST | `/api/simulations` | 시뮬레이션 작업 생성 |
| GET | `/api/simulations/{job_id}` | 작업 상태와 결과 조회 |

`/api/policies/direct`도 공통 페르소나 필터와 시뮬레이션 검증 경로를 사용합니다.

## 데이터와 제출 경계

현재 제품에 포함해야 하는 데이터:

```text
data/raw/policies/
data/raw/faq/civil_policy_qna_detail.json
data/raw/faq/civil_policy_qna_metadata.json
data/indexes/policies/current/
data/indexes/civil_complaints/current/active.json
data/indexes/civil_complaints/versions/<active-version>/
```

서버에서 필요하지만 Git에는 포함하지 않는 생성·캐시 데이터:

```text
data/raw/personas/
data/runtime/
.venv/
frontend/node_modules/
frontend/dist/
tmp/
```

연구용 법령 원천, 학습 체크포인트, 실험 출력, 연구 스크립트와 구형 RAG
프로토타입은 제품 실행 경계에 포함하지 않습니다.

제출 전 런타임 자산과 연구 잔존물을 한 번에 검사할 수 있습니다.

```bash
.venv/bin/python scripts/verify_submission.py
```

## 정책·인덱스 갱신

일상적인 서비스 실행에는 환경변수가 필요하지 않습니다. 정부24 정책 원천을
갱신할 때만 공공데이터포털 인증키가 필요합니다.

```bash
cp .env.example .env
# .env의 GOV24_OPENAPI_SERVICE_KEY 값을 입력합니다.

.venv/bin/python scripts/data/fetch_gov24_policy_openapi.py --dry-run
.venv/bin/python scripts/data/fetch_gov24_policy_openapi.py
.venv/bin/python scripts/rag/build_policy_index.py
```

공개 민원 FAQ 정본을 변경한 경우 민원 인덱스도 다시 생성합니다.

```bash
.venv/bin/python scripts/rag/build_civil_complaint_index.py
```

두 인덱스 빌더는 검증된 새 인덱스를 만든 뒤 원자적으로 활성화합니다. 갱신 후
`scripts/verify_submission.py`와 전체 테스트를 통과한 다음 이전 백업·비활성 버전을
프로젝트 밖으로 이동합니다.

## 품질 검사

백엔드 제품 회귀 테스트:

```bash
.venv/bin/python -m unittest discover -s backend/tests -p 'test_*.py' -v
```

파이썬 구문, 패치 공백, 프런트 정적 검사와 빌드:

```bash
.venv/bin/python -m compileall -q backend scripts
git diff --check

cd frontend
npm run lint
npm run build
```

최종 제출 판정은 위 자동 검사뿐 아니라, 동일한 최종 소스 해시에서 실제 Qwen 시민
시뮬레이션 10회를 연속 실행해 시민 30명과 그에 연결된 정책 근거 공무원 답변 30개,
지역·나이·품질 게이트와 GPU 정리 상태까지 확인해야 합니다.

## 저장소 구조

```text
backend/
  api.py                         FastAPI 앱
  ai_simulation_core/            정책·페르소나·생성·검증·검색
  tests/                         제품 회귀 테스트
frontend/
  src/                           React 화면과 컴포넌트
data/
  raw/policies/                  정부24 정책 정본
  raw/faq/                       공개 민원 FAQ 정본
  indexes/                       활성 검색 인덱스
  runtime/                       실행 중 생성되는 상태
scripts/
  data/                          정책 원천 갱신
  rag/                           정책·민원 인덱스 생성
  verify_submission.py           제출 경계 검증
```

## 라이선스 참고

사용 중인 Pretendard GOV 폰트의 라이선스는
`frontend/src/assets/fonts/LICENSE.md`에 포함되어 있습니다. 프로젝트 전체 배포
라이선스는 제출처의 요구사항과 소유권 정책에 맞춰 별도로 결정해야 합니다.
