import { useEffect, useState } from "react";
import { ArrowLeft, Check, MapPin, RefreshCw, Shuffle, User } from "lucide-react";

const MAX_SELECTED_PERSONAS = 3;
const CANDIDATE_LIMIT = 12;
const AGE_MINIMUM = 0;
const AGE_MAXIMUM = 120;

function getApiError(result, fallbackMessage) {
  if (typeof result?.detail === "string") return result.detail;

  if (Array.isArray(result?.detail)) {
    const messages = result.detail
      .map((item) => item?.msg)
      .filter((message) => typeof message === "string" && message.trim());
    if (messages.length > 0) return messages.join(" ");
  }

  return fallbackMessage;
}

function parseOptionalAge(value) {
  if (value === "") return null;
  const parsed = Number(value);
  return Number.isInteger(parsed) ? parsed : null;
}

function validateAgeRange(ageMin, ageMax) {
  const values = [ageMin, ageMax].filter((value) => value !== "");
  const hasOutOfRangeValue = values.some((value) => {
    const parsed = Number(value);
    return (
      !Number.isInteger(parsed) ||
      parsed < AGE_MINIMUM ||
      parsed > AGE_MAXIMUM
    );
  });

  if (hasOutOfRangeValue) {
    return `나이는 ${AGE_MINIMUM}세부터 ${AGE_MAXIMUM}세 사이의 정수로 입력해 주세요.`;
  }

  if (ageMin !== "" && ageMax !== "" && Number(ageMin) > Number(ageMax)) {
    return "최소 나이는 최대 나이보다 클 수 없습니다.";
  }

  return "";
}

function displayDistrict(district, province) {
  if (!district) return "";
  const exactPrefix = province ? `${province}-` : "";
  if (exactPrefix && district.startsWith(exactPrefix)) {
    return district.slice(exactPrefix.length);
  }

  const separatorIndex = district.indexOf("-");
  return separatorIndex >= 0 ? district.slice(separatorIndex + 1) : district;
}

function formatAgeMatchReason(reason, ageMin, ageMax) {
  if (reason === "within_range") {
    if (ageMin !== null && ageMax !== null) {
      return `${ageMin}~${ageMax}세 범위 충족`;
    }
    if (ageMin !== null) return `${ageMin}세 이상 조건 충족`;
    if (ageMax !== null) return `${ageMax}세 이하 조건 충족`;
    return "연령 제한 없음";
  }
  if (reason === "below_minimum") {
    return ageMin === null ? "연령 하한 경계" : `${ageMin}세 하한 바로 아래`;
  }
  if (reason === "above_maximum") {
    return ageMax === null ? "연령 상한 경계" : `${ageMax}세 상한 바로 위`;
  }
  return "연령 조건 확인";
}

function CandidateCard({
  candidate,
  isSelected,
  isDisabled,
  ageMin,
  ageMax,
  onToggle,
}) {
  const location = [candidate.province, displayDistrict(candidate.district, candidate.province)]
    .filter(Boolean)
    .join(" ");
  const ageReason = formatAgeMatchReason(
    candidate.match?.age_match_reason,
    ageMin,
    ageMax,
  );
  const cohortLabel =
    candidate.match?.age_cohort === "boundary" ? "연령 경계선" : "정책 해당";

  return (
    <button
      type="button"
      aria-pressed={isSelected}
      disabled={isDisabled}
      onClick={() => onToggle(candidate)}
      className={`group relative flex min-h-[246px] flex-col rounded-[20px] border p-5 text-left transition focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 ${
        isSelected
          ? "border-brand bg-brand-soft/65 shadow-[0_10px_24px_rgba(44,74,110,0.12)]"
          : "border-[#dde3ea] bg-white hover:border-[#9cabbc] hover:shadow-[0_10px_24px_rgba(20,23,28,0.06)]"
      }`}
    >
      <span
        className={`absolute right-4 top-4 flex h-7 w-7 items-center justify-center rounded-full border transition ${
          isSelected
            ? "border-brand bg-brand text-white"
            : "border-[#cbd3dc] bg-white text-transparent group-hover:border-brand"
        }`}
        aria-hidden="true"
      >
        <Check size={16} strokeWidth={2.3} />
      </span>

      <div className="flex items-center gap-3 pr-10">
        <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#eef1f5] text-brand">
          <User aria-hidden="true" size={22} strokeWidth={1.8} />
        </span>
        <div className="min-w-0">
          <p className="truncate text-[17px] font-bold text-ink">
            {candidate.occupation || "직업 정보 없음"}
          </p>
          <p className="mt-0.5 text-[13px] text-slate">
            {[candidate.age !== null && candidate.age !== undefined
              ? `${candidate.age}세`
              : null, candidate.sex]
              .filter(Boolean)
              .join(" · ") || "기본 정보 없음"}
          </p>
        </div>
      </div>

      <p className="mt-4 flex items-center gap-1.5 text-[13px] text-slate">
        <MapPin aria-hidden="true" size={15} strokeWidth={1.8} />
        {location || "지역 정보 없음"}
      </p>

      <p className="mt-3 max-h-[48px] overflow-hidden text-[13px] leading-6 text-slate">
        {candidate.persona || "페르소나 성향 정보가 없습니다."}
      </p>

      <div className="mt-auto flex flex-wrap gap-2 pt-4">
        <span className="rounded-pill bg-white px-2.5 py-1 text-[11px] font-semibold text-brand ring-1 ring-[#cfd8e3]">
          {cohortLabel}
        </span>
        <span className="rounded-pill bg-white px-2.5 py-1 text-[11px] font-medium text-slate ring-1 ring-[#dce1e7]">
          {candidate.match?.region_match === false ? "지역 불일치" : "정책 지역 일치"}
        </span>
        <span className="rounded-pill bg-white px-2.5 py-1 text-[11px] font-medium text-slate ring-1 ring-[#dce1e7]">
          {ageReason}
        </span>
      </div>
    </button>
  );
}

export default function PersonaSelectionStep({
  policyDraft,
  onBack,
  onSimulationStarted,
}) {
  const [regionScope, setRegionScope] = useState("specific");
  const [province, setProvince] = useState("");
  const [district, setDistrict] = useState("");
  const [ageMin, setAgeMin] = useState("");
  const [ageMax, setAgeMax] = useState("");
  const [cohort, setCohort] = useState("eligible");
  const [regionOptions, setRegionOptions] = useState([]);
  const [optionsState, setOptionsState] = useState("loading");
  const [optionsError, setOptionsError] = useState("");
  const [candidates, setCandidates] = useState([]);
  const [candidateState, setCandidateState] = useState("idle");
  const [candidateError, setCandidateError] = useState("");
  const [selectedPersonas, setSelectedPersonas] = useState(() => new Map());
  const [refreshSeed, setRefreshSeed] = useState(0);
  const [submissionState, setSubmissionState] = useState("idle");
  const [submissionMode, setSubmissionMode] = useState(null);
  const [submissionError, setSubmissionError] = useState("");

  const selectedProvince = regionOptions.find(
    (option) => option.province === province,
  );
  const districtOptions = selectedProvince?.districts ?? [];
  const ageError = validateAgeRange(ageMin, ageMax);
  const hasAgeCondition = ageMin !== "" || ageMax !== "";
  const hasRequiredRegion = regionScope === "nationwide" || Boolean(province);
  const filtersAreValid = hasRequiredRegion && !ageError;
  const isLoadingCandidates = candidateState === "loading";
  const isSubmitting = submissionState === "submitting";
  const selectedIds = [...selectedPersonas.keys()];
  const hasThreeSelections = selectedIds.length === MAX_SELECTED_PERSONAS;

  useEffect(() => {
    const controller = new AbortController();

    async function loadRegionOptions() {
      setOptionsState("loading");
      setOptionsError("");

      try {
        const response = await fetch("/api/personas/options", {
          signal: controller.signal,
        });
        const result = await response.json().catch(() => null);

        if (!response.ok) {
          throw new Error(
            getApiError(result, "지역 선택 정보를 불러오지 못했습니다."),
          );
        }

        setRegionOptions(Array.isArray(result?.provinces) ? result.provinces : []);
        setOptionsState("success");
      } catch (error) {
        if (error?.name === "AbortError") return;
        setOptionsState("error");
        setOptionsError(
          error instanceof Error
            ? error.message
            : "지역 선택 정보를 불러오지 못했습니다.",
        );
      }
    }

    void loadRegionOptions();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!filtersAreValid || (cohort === "boundary" && !hasAgeCondition)) {
      setCandidates([]);
      setCandidateState("idle");
      setCandidateError("");
      return undefined;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      const params = new URLSearchParams({
        region_scope: regionScope,
        cohort,
        limit: String(CANDIDATE_LIMIT),
        seed: String(refreshSeed),
      });

      if (regionScope === "specific") {
        params.set("province", province);
        if (district) params.set("district", district);
      }
      if (ageMin !== "") params.set("age_min", ageMin);
      if (ageMax !== "") params.set("age_max", ageMax);

      setCandidateState("loading");
      setCandidateError("");
      setCandidates([]);

      try {
        const response = await fetch(`/api/personas/candidates?${params}`, {
          signal: controller.signal,
        });
        const result = await response.json().catch(() => null);

        if (!response.ok) {
          throw new Error(
            getApiError(result, "페르소나 후보를 불러오지 못했습니다."),
          );
        }

        // 후보 응답이 배열이 아니면 빈 배열로 정규화해 렌더링 계약을 고정한다.
        setCandidates(Array.isArray(result?.candidates) ? result.candidates : []);
        setCandidateState("success");
      } catch (error) {
        if (error?.name === "AbortError") return;
        setCandidateState("error");
        setCandidateError(
          error instanceof Error
            ? error.message
            : "페르소나 후보를 불러오지 못했습니다.",
        );
      }
    }, 200);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [
    ageMax,
    ageMin,
    cohort,
    district,
    filtersAreValid,
    hasAgeCondition,
    province,
    refreshSeed,
    regionScope,
  ]);

  const clearSelectionForFilterChange = () => {
    setSelectedPersonas(new Map());
    setSubmissionState("idle");
    setSubmissionMode(null);
    setSubmissionError("");
  };

  const changeRegionScope = (nextScope) => {
    if (nextScope === regionScope) return;
    setRegionScope(nextScope);
    setProvince("");
    setDistrict("");
    clearSelectionForFilterChange();
  };

  const changeProvince = (event) => {
    setProvince(event.target.value);
    setDistrict("");
    clearSelectionForFilterChange();
  };

  const changeDistrict = (event) => {
    setDistrict(event.target.value);
    clearSelectionForFilterChange();
  };

  const changeAgeMin = (event) => {
    const nextAgeMin = event.target.value;
    setAgeMin(nextAgeMin);
    if (cohort === "boundary" && nextAgeMin === "" && ageMax === "") {
      setCohort("eligible");
    }
    clearSelectionForFilterChange();
  };

  const changeAgeMax = (event) => {
    const nextAgeMax = event.target.value;
    setAgeMax(nextAgeMax);
    if (cohort === "boundary" && ageMin === "" && nextAgeMax === "") {
      setCohort("eligible");
    }
    clearSelectionForFilterChange();
  };

  const changeCohort = (nextCohort) => {
    if (nextCohort === cohort || (nextCohort === "boundary" && !hasAgeCondition)) {
      return;
    }
    setCohort(nextCohort);
    setCandidates([]);
    setCandidateState("loading");
    setSubmissionState("idle");
    setSubmissionMode(null);
    setSubmissionError("");
  };

  const refreshCandidates = () => {
    setRefreshSeed((current) => current + 1);
    setCandidates([]);
    setCandidateState("loading");
    setSubmissionState("idle");
    setSubmissionMode(null);
    setSubmissionError("");
  };

  const toggleCandidate = (candidate) => {
    setSelectedPersonas((currentPersonas) => {
      const nextPersonas = new Map(currentPersonas);
      if (nextPersonas.has(candidate.uuid)) {
        nextPersonas.delete(candidate.uuid);
        return nextPersonas;
      }
      if (nextPersonas.size >= MAX_SELECTED_PERSONAS) return currentPersonas;
      nextPersonas.set(candidate.uuid, candidate);
      return nextPersonas;
    });
    setSubmissionState("idle");
    setSubmissionMode(null);
    setSubmissionError("");
  };

  const submitSimulation = async (selectionMode) => {
    const isRandomSelection = selectionMode === "random";
    if (
      isSubmitting ||
      !filtersAreValid ||
      (!isRandomSelection && !hasThreeSelections)
    ) {
      return;
    }

    setSubmissionState("submitting");
    setSubmissionMode(selectionMode);
    setSubmissionError("");

    const policy = {
      ...policyDraft,
      region_scope: regionScope,
      region_province: regionScope === "specific" ? province : "",
      region_district: regionScope === "specific" ? district : "",
      age_min: parseOptionalAge(ageMin),
      age_max: parseOptionalAge(ageMax),
      age_basis: "dataset_age",
    };
    // random은 ID 없이 서버가 eligible 3명을 선택하고,
    // manual은 선택한 3개 ID의 지역·연령 cohort를 서버가 다시 검증한다.
    const requestBody = {
      policy,
      selection_mode: selectionMode,
      ...(isRandomSelection ? {} : { persona_ids: selectedIds }),
    };
    const fallbackMessage = isRandomSelection
      ? "범위에 맞는 무작위 페르소나로 분석을 시작하지 못했습니다."
      : "선택한 페르소나로 분석을 시작하지 못했습니다.";

    try {
      const response = await fetch("/api/simulations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      const result = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(getApiError(result, fallbackMessage));
      }

      if (!result?.job_id || !result?.policy) {
        throw new Error("시뮬레이션 작업 ID를 받지 못했습니다.");
      }

      onSimulationStarted(result);
    } catch (error) {
      setSubmissionState("error");
      setSubmissionError(
        error instanceof Error ? error.message : fallbackMessage,
      );
    }
  };

  const selectionStatus = `${selectedPersonas.size}/${MAX_SELECTED_PERSONAS}명 선택`;

  return (
    <main className="min-h-[calc(100vh-76px)] bg-[#f6f7fb] px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <div className="mx-auto max-w-[1180px]">
        <button
          type="button"
          onClick={onBack}
          disabled={isSubmitting}
          className="inline-flex items-center gap-2 rounded-[10px] px-2 py-2 text-[14px] font-semibold text-brand transition hover:bg-brand-soft disabled:cursor-not-allowed disabled:opacity-50"
        >
          <ArrowLeft aria-hidden="true" size={18} strokeWidth={1.9} />
          정책 내용으로 돌아가기
        </button>

        <div className="mt-4 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="text-[13px] font-semibold text-brand">2단계 · 분석 대상 설정</p>
            <h1 className="mt-1 text-[30px] font-bold tracking-[-0.025em] text-ink sm:text-[34px]">
              페르소나 선택
            </h1>
            <p className="mt-2 text-[15px] leading-6 text-slate">
              직접 3명을 선택하거나, 현재 지역·나이 범위에 맞는 3명을 무작위로 선택해 분석하세요.
            </p>
          </div>
          <div className="rounded-pill bg-brand-soft px-4 py-2 text-[14px] font-bold text-brand" aria-live="polite">
            {selectionStatus}
          </div>
        </div>

        <section className="mt-7 rounded-[22px] border border-[#dde1e8] bg-white p-6 shadow-[0_12px_30px_rgba(20,23,28,0.05)] sm:p-7">
          <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr_1fr]">
            <fieldset>
              <legend className="text-[14px] font-bold text-ink">지역 범위</legend>
              <div className="mt-2 grid grid-cols-2 rounded-[12px] bg-[#f1f3f6] p-1">
                <button
                  type="button"
                  aria-pressed={regionScope === "nationwide"}
                  disabled={isSubmitting}
                  onClick={() => changeRegionScope("nationwide")}
                  className={`rounded-[9px] px-3 py-2.5 text-[14px] font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                    regionScope === "nationwide"
                      ? "bg-white text-brand shadow-sm"
                      : "text-slate hover:text-ink"
                  }`}
                >
                  전국
                </button>
                <button
                  type="button"
                  aria-pressed={regionScope === "specific"}
                  disabled={isSubmitting}
                  onClick={() => changeRegionScope("specific")}
                  className={`rounded-[9px] px-3 py-2.5 text-[14px] font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                    regionScope === "specific"
                      ? "bg-white text-brand shadow-sm"
                      : "text-slate hover:text-ink"
                  }`}
                >
                  특정 지역
                </button>
              </div>
            </fieldset>

            <div className="grid gap-4 sm:grid-cols-2">
              <label className="text-[14px] font-bold text-ink">
                시·도
                <select
                  value={province}
                  onChange={changeProvince}
                  disabled={
                    isSubmitting ||
                    regionScope === "nationwide" ||
                    optionsState === "loading"
                  }
                  className="mt-2 w-full rounded-[12px] border border-[#d8dde5] bg-white px-3 py-3 text-[14px] font-medium text-ink focus:border-brand focus:outline-none disabled:cursor-not-allowed disabled:bg-[#f1f3f5] disabled:text-slate"
                >
                  <option value="">
                    {optionsState === "loading" ? "불러오는 중..." : "시·도 선택"}
                  </option>
                  {regionOptions.map((option) => (
                    <option key={option.province} value={option.province}>
                      {option.province}
                    </option>
                  ))}
                </select>
              </label>

              <label className="text-[14px] font-bold text-ink">
                시·군·구 <span className="font-medium text-slate">(선택)</span>
                <select
                  value={district}
                  onChange={changeDistrict}
                  disabled={isSubmitting || regionScope === "nationwide" || !province}
                  className="mt-2 w-full rounded-[12px] border border-[#d8dde5] bg-white px-3 py-3 text-[14px] font-medium text-ink focus:border-brand focus:outline-none disabled:cursor-not-allowed disabled:bg-[#f1f3f5] disabled:text-slate"
                >
                  <option value="">시·도 전체</option>
                  {districtOptions.map((option) => (
                    <option key={option} value={option}>
                      {displayDistrict(option, province)}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <fieldset>
              <legend className="text-[14px] font-bold text-ink">표기 나이 범위</legend>
              <div className="mt-2 grid grid-cols-[1fr_auto_1fr] items-center gap-2">
                <label className="sr-only" htmlFor="persona-age-min">
                  최소 표기 나이
                </label>
                <input
                  id="persona-age-min"
                  type="number"
                  inputMode="numeric"
                  min={AGE_MINIMUM}
                  max={AGE_MAXIMUM}
                  value={ageMin}
                  onChange={changeAgeMin}
                  disabled={isSubmitting}
                  placeholder="최소"
                  className="min-w-0 rounded-[12px] border border-[#d8dde5] bg-white px-3 py-3 text-[14px] text-ink placeholder:text-slate/60 focus:border-brand focus:outline-none"
                />
                <span className="text-[13px] text-slate">~</span>
                <label className="sr-only" htmlFor="persona-age-max">
                  최대 표기 나이
                </label>
                <input
                  id="persona-age-max"
                  type="number"
                  inputMode="numeric"
                  min={AGE_MINIMUM}
                  max={AGE_MAXIMUM}
                  value={ageMax}
                  onChange={changeAgeMax}
                  disabled={isSubmitting}
                  placeholder="최대"
                  className="min-w-0 rounded-[12px] border border-[#d8dde5] bg-white px-3 py-3 text-[14px] text-ink placeholder:text-slate/60 focus:border-brand focus:outline-none"
                />
              </div>
              <p className="mt-2 text-[12px] leading-5 text-slate">
                생년월일이 아닌 데이터셋의 정수 표기 나이를 기준으로 판정합니다.
              </p>
            </fieldset>
          </div>

          {optionsState === "error" && (
            <p role="alert" className="mt-4 text-[13px] font-medium text-[#9a342b]">
              {optionsError}
            </p>
          )}
          {ageError && (
            <p role="alert" className="mt-4 text-[13px] font-medium text-[#9a342b]">
              {ageError}
            </p>
          )}
          {regionScope === "specific" && province && (
            <p className="mt-5 rounded-[13px] bg-brand-soft/75 px-4 py-3 text-[13px] leading-5 text-brand">
              선택한 {province}
              {district ? ` ${displayDistrict(district, province)}` : " 전체"} 거주 페르소나만 후보에 표시됩니다.
            </p>
          )}
        </section>

        <section className="mt-6" aria-labelledby="candidate-heading">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 id="candidate-heading" className="text-[22px] font-bold text-ink">
                페르소나 후보
              </h2>
              <p className="mt-1 text-[13px] text-slate">
                경계선 후보는 입력한 최소·최대 나이의 바로 바깥 연령을 보여줍니다.
              </p>
            </div>
            <button
              type="button"
              onClick={refreshCandidates}
              disabled={!filtersAreValid || isLoadingCandidates || isSubmitting}
              className="inline-flex items-center gap-2 rounded-[11px] border border-[#cfd6df] bg-white px-4 py-2.5 text-[13px] font-semibold text-brand transition hover:border-brand hover:bg-brand-soft disabled:cursor-not-allowed disabled:opacity-50"
            >
              <RefreshCw
                aria-hidden="true"
                size={16}
                strokeWidth={1.9}
                className={isLoadingCandidates ? "animate-spin" : ""}
              />
              후보 새로고침
            </button>
          </div>

          <div className="mt-5 flex w-full max-w-[420px] rounded-[13px] bg-[#e9ecf0] p-1" role="group" aria-label="페르소나 후보 유형">
            <button
              type="button"
              aria-pressed={cohort === "eligible"}
              disabled={isSubmitting}
              onClick={() => changeCohort("eligible")}
              className={`flex-1 rounded-[10px] px-4 py-2.5 text-[14px] font-semibold transition disabled:cursor-not-allowed disabled:opacity-45 ${
                cohort === "eligible"
                  ? "bg-white text-brand shadow-sm"
                  : "text-slate hover:text-ink"
              }`}
            >
              정책 해당
            </button>
            <button
              type="button"
              aria-pressed={cohort === "boundary"}
              disabled={isSubmitting || !hasAgeCondition || Boolean(ageError)}
              onClick={() => changeCohort("boundary")}
              className={`flex-1 rounded-[10px] px-4 py-2.5 text-[14px] font-semibold transition disabled:cursor-not-allowed disabled:opacity-45 ${
                cohort === "boundary"
                  ? "bg-white text-brand shadow-sm"
                  : "text-slate hover:text-ink"
              }`}
            >
              연령 경계선
            </button>
          </div>

          {!hasRequiredRegion ? (
            <div className="mt-5 rounded-[18px] border border-dashed border-[#cfd6df] bg-white px-6 py-12 text-center">
              <p className="text-[15px] font-semibold text-ink">정책 적용 지역을 선택해 주세요.</p>
              <p className="mt-2 text-[13px] text-slate">
                특정 지역을 선택하면 다른 지역의 페르소나는 후보에 포함되지 않습니다.
              </p>
            </div>
          ) : isLoadingCandidates ? (
            <div className="mt-5 rounded-[18px] border border-[#dde1e8] bg-white px-6 py-12 text-center" role="status">
              <RefreshCw aria-hidden="true" size={23} className="mx-auto animate-spin text-brand" />
              <p className="mt-3 text-[14px] font-medium text-slate">조건에 맞는 후보를 불러오는 중입니다.</p>
            </div>
          ) : candidateState === "error" ? (
            <div className="mt-5 rounded-[18px] border border-[#ead1ce] bg-[#fffafa] px-6 py-9 text-center">
              <p role="alert" className="text-[14px] font-semibold text-[#9a342b]">
                {candidateError}
              </p>
              <button
                type="button"
                onClick={refreshCandidates}
                className="mt-4 rounded-[10px] bg-brand px-4 py-2.5 text-[13px] font-semibold text-white hover:bg-brand-strong"
              >
                다시 시도
              </button>
            </div>
          ) : candidateState === "success" && candidates.length === 0 ? (
            <div className="mt-5 rounded-[18px] border border-dashed border-[#cfd6df] bg-white px-6 py-12 text-center">
              <p className="text-[15px] font-semibold text-ink">조건에 맞는 후보가 없습니다.</p>
              <p className="mt-2 text-[13px] text-slate">
                지역 또는 나이 범위를 조정한 뒤 다시 확인해 주세요.
              </p>
            </div>
          ) : (
            <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {candidates.map((candidate) => {
                const isSelected = selectedPersonas.has(candidate.uuid);
                return (
                  <CandidateCard
                    key={candidate.uuid}
                    candidate={candidate}
                    isSelected={isSelected}
                    isDisabled={
                      isSubmitting ||
                      (!isSelected &&
                        selectedPersonas.size >= MAX_SELECTED_PERSONAS)
                    }
                    ageMin={parseOptionalAge(ageMin)}
                    ageMax={parseOptionalAge(ageMax)}
                    onToggle={toggleCandidate}
                  />
                );
              })}
            </div>
          )}
        </section>

        <div className="sticky bottom-2 z-10 mt-8 flex flex-wrap items-center justify-between gap-3 rounded-[18px] border border-[#d9dfe6] bg-white/95 px-4 py-3 shadow-[0_14px_36px_rgba(20,23,28,0.13)] backdrop-blur sm:bottom-4 sm:gap-4 sm:px-6 sm:py-4">
          <div>
            <p className="text-[15px] font-bold text-ink">{selectionStatus}</p>
            <p className="mt-0.5 hidden max-w-[510px] text-[12px] leading-5 text-slate sm:block">
              직접 선택하지 않아도 현재 지역·나이의 정책 해당 범위에서 서버가 서로 다른 3명을 무작위로 선택합니다.
              연령 경계선 페르소나는 직접 선택할 때만 포함됩니다.
            </p>
            {selectedPersonas.size > 0 && (
              <ul className="mt-2 flex flex-wrap gap-2" aria-label="선택한 페르소나">
                {[...selectedPersonas.values()].map((candidate) => (
                  <li key={candidate.uuid}>
                    <button
                      type="button"
                      disabled={isSubmitting}
                      onClick={() => toggleCandidate(candidate)}
                      aria-label={`${candidate.occupation || "직업 정보 없음"} 선택 해제`}
                      className="rounded-pill bg-brand-soft px-2.5 py-1 text-[11px] font-semibold text-brand transition hover:bg-[#dce5ee] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {candidate.occupation || "직업 정보 없음"} · {candidate.age ?? "나이 미상"}
                      {candidate.age !== null && candidate.age !== undefined ? "세" : ""} · {candidate.match?.age_cohort === "boundary" ? "연령 경계선" : "정책 해당"}
                      <span className="ml-1.5" aria-hidden="true">×</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="flex w-full flex-col items-stretch justify-end gap-2.5 sm:w-auto sm:flex-1 sm:flex-row sm:flex-wrap sm:items-center sm:gap-3 lg:flex-none">
            {submissionState === "error" && (
              <p role="alert" className="max-w-[420px] text-[13px] font-medium text-[#9a342b]">
                {submissionError}
              </p>
            )}
            <button
              type="button"
              onClick={() => submitSimulation("random")}
              disabled={!filtersAreValid || isSubmitting}
              className="inline-flex w-full items-center justify-center gap-2 rounded-[13px] border border-[#b7c4d2] bg-white px-4 py-3 text-[14px] font-semibold text-brand transition hover:border-brand hover:bg-brand-soft disabled:cursor-not-allowed disabled:border-[#d7dce2] disabled:bg-[#eef1f4] disabled:text-[#9ba6b3] sm:w-auto sm:min-w-[230px] sm:px-5 sm:py-3.5 sm:text-[15px]"
            >
              <Shuffle aria-hidden="true" size={17} strokeWidth={2} />
              {isSubmitting && submissionMode === "random"
                ? "무작위 3명 선택 중..."
                : (
                    <>
                      <span className="sm:hidden">무작위 3명으로 분석</span>
                      <span className="hidden sm:inline">범위 내 무작위 3명으로 분석</span>
                    </>
                  )}
            </button>
            <button
              type="button"
              onClick={() => submitSimulation("manual")}
              disabled={!hasThreeSelections || isSubmitting || !filtersAreValid}
              className="w-full rounded-[13px] bg-brand px-4 py-3 text-[14px] font-semibold text-white shadow-[0_8px_18px_rgba(44,74,110,0.18)] transition hover:bg-brand-strong disabled:cursor-not-allowed disabled:bg-[#b8c1cc] disabled:shadow-none sm:w-auto sm:min-w-[230px] sm:px-6 sm:py-3.5 sm:text-[15px]"
            >
              {isSubmitting && submissionMode === "manual"
                ? "분석 시작 중..."
                : (
                    <>
                      <span className="sm:hidden">선택한 3명으로 분석</span>
                      <span className="hidden sm:inline">선택한 페르소나로 분석 시작</span>
                    </>
                  )}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
