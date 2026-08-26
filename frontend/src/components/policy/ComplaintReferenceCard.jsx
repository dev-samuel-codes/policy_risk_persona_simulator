import { useId } from "react";

const EVIDENCE_FIELD_LABELS = {
  age: "연령 조건",
  age_condition: "연령 조건",
  complaint_type: "민원 유형",
  domain_tags: "정책 분야",
  eligibility: "자격",
  issue_tags: "민원 유형",
  policy_field: "정책 분야",
  policy_region: "정책 지역",
  qualification: "자격",
  qualification_tags: "자격",
  region: "정책 지역",
};

function firstPresentValue(source, keys) {
  for (const key of keys) {
    const value = source?.[key];
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return null;
}

function isConfirmedEvidence(item) {
  if (!item || typeof item !== "object") return true;
  return !(
    item.confirmed === false ||
    item.matched === false ||
    item.reference_eligible === false ||
    item.verified === false
  );
}

function formatDetailText(detail) {
  const values = Array.isArray(detail) ? detail : [detail];

  return values
    .map((item) => {
      if (typeof item === "string") return item.trim();
      if (item && typeof item === "object") {
        return firstPresentValue(item, [
          "label",
          "description",
          "detail",
          "value",
        ]);
      }
      return item === null || item === undefined ? null : String(item);
    })
    .filter(Boolean)
    .join(" · ");
}

function formatEvidence(field, evidence) {
  if (evidence === false || evidence === null || evidence === undefined) {
    return null;
  }

  if (typeof evidence === "string") {
    return evidence.trim() || null;
  }

  const fieldLabel = EVIDENCE_FIELD_LABELS[field] ?? field;
  if (evidence === true) return fieldLabel;
  if (!isConfirmedEvidence(evidence)) return null;

  const label = firstPresentValue(evidence, ["label", "field_label"]) ?? fieldLabel;
  const detail = firstPresentValue(evidence, [
    "reason",
    "description",
    "detail",
    "details",
    "value",
    "matched_value",
  ]);
  const detailText = formatDetailText(detail);

  if (!detailText || detailText === label) return label;
  return `${label}: ${detailText}`;
}

function normalizeEvidence(referenceCase) {
  const matchReasons = referenceCase?.match_reasons;
  const hasMatchReasons = Array.isArray(matchReasons)
    ? matchReasons.length > 0
    : Boolean(matchReasons && Object.keys(matchReasons).length > 0);
  const source = hasMatchReasons ? matchReasons : referenceCase?.evidence;
  if (!source) return [];

  if (Array.isArray(source)) {
    return source
      .map((item, index) => {
        if (typeof item === "string") {
          const label = item.trim();
          return label ? { key: `${index}-${label}`, label } : null;
        }
        if (!isConfirmedEvidence(item)) return null;

        const field = firstPresentValue(item, ["field", "type", "key"]);
        const label = formatEvidence(field ?? `근거 ${index + 1}`, item);
        return label ? { key: `${field ?? index}-${label}`, label } : null;
      })
      .filter(Boolean)
      .slice(0, 5);
  }

  if (typeof source === "object") {
    return Object.entries(source)
      .map(([field, evidence]) => {
        const label = formatEvidence(field, evidence);
        return label ? { key: field, label } : null;
      })
      .filter(Boolean)
      .slice(0, 5);
  }

  return [];
}

function referenceSearchStatus(complaint) {
  const search = complaint?.precedent_search;
  return typeof search === "string" ? search : search?.status;
}

function ReferenceSearchMessage({ status }) {
  if (status === "no_reliable_match") {
    return (
      <p
        role="status"
        className="mt-3 rounded-[16px] border border-[#dde3ea] bg-[#f7f8fa] px-4 py-3 text-[12px] leading-5 text-slate"
      >
        현재 지역·연령·정책 분야 기준으로 표시할 참고 사례를 찾지 못했습니다.
      </p>
    );
  }

  if (status === "unavailable") {
    return (
      <p
        role="alert"
        className="mt-3 rounded-[16px] border border-[#ead9bb] bg-[#fffaf0] px-4 py-3 text-[12px] leading-5 text-[#715523]"
      >
        공개 민원 Q&amp;A 참고 사례 검색을 완료하지 못했습니다. 시뮬레이션 결과에는
        영향을 주지 않습니다.
      </p>
    );
  }

  if (status === "invalid_query") {
    return (
      <p
        role="status"
        className="mt-3 rounded-[16px] border border-[#dde3ea] bg-[#f7f8fa] px-4 py-3 text-[12px] leading-5 text-slate"
      >
        근거 검색에 필요한 민원 요약이 부족하여 참고 사례를 표시하지 않았습니다.
      </p>
    );
  }

  return null;
}

function formatRelatedLaws(value) {
  const values = Array.isArray(value) ? value : [value];
  const labels = values
    .map((item) => {
      if (typeof item === "string") return item.trim();
      if (item && typeof item === "object") {
        return firstPresentValue(item, ["full_name", "name", "title"]);
      }
      return null;
    })
    .filter(Boolean);

  return labels.length > 0 ? labels.join(" · ") : null;
}

function formatMatchScore(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toLocaleString("ko-KR", { maximumFractionDigits: 1 });
  }
  if (typeof value === "string" && value.trim()) return value.trim();
  return null;
}

function getReferenceCaseDetails(referenceCase) {
  return {
    date: firstPresentValue(referenceCase, [
      "registered_at",
      "published_at",
      "date",
    ]),
    laws: formatRelatedLaws(
      firstPresentValue(referenceCase, [
        "related_laws",
        "related_law",
        "law",
      ]),
    ),
    organization: firstPresentValue(referenceCase, [
      "organization",
      "agency",
      "institution",
    ]),
    score: formatMatchScore(
      firstPresentValue(referenceCase, [
        "match_score",
        "search_similarity_score",
        "similarity_score",
        "score",
      ]),
    ),
    title: firstPresentValue(referenceCase, ["title", "subject"]),
  };
}

export function ComplaintReferenceCard({ complaint }) {
  const descriptionId = useId();
  const status = referenceSearchStatus(complaint);
  const referenceCase = complaint?.reference_cases?.find(
    (candidate) => candidate?.reference_eligible === true,
  );

  if (status !== "matched") {
    return <ReferenceSearchMessage status={status} />;
  }

  if (!referenceCase) {
    return <ReferenceSearchMessage status="no_reliable_match" />;
  }

  const { date, laws, organization, score, title } =
    getReferenceCaseDetails(referenceCase);
  const evidence = normalizeEvidence(referenceCase);

  return (
    <article
      aria-describedby={descriptionId}
      className="mt-3 rounded-[16px] border border-[#dde3ea] bg-[#f7f8fa] px-4 py-3"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <p className="text-[11px] font-semibold leading-5 text-slate">
          조건이 확인된 공개 민원 Q&amp;A 참고 사례
        </p>
        {score !== null && (
          <span className="rounded-pill bg-brand-soft px-2.5 py-1 text-[11px] font-bold text-brand">
            검색 유사도 점수 {score}
          </span>
        )}
      </div>

      <h3 className="mt-2 text-[13px] font-semibold leading-5 text-ink">
        {title ?? "공개 민원 Q&A 참고 사례"}
      </h3>

      {(organization || date) && (
        <p className="mt-1 text-[11px] leading-5 text-slate">
          {organization}
          {organization && date ? " · " : null}
          {date ? <time dateTime={date}>{date}</time> : null}
        </p>
      )}

      {laws && (
        <p className="mt-1 text-[11px] leading-5 text-slate/80">{laws}</p>
      )}

      {evidence.length > 0 && (
        <div className="mt-3">
          <p className="text-[11px] font-semibold text-slate">확인된 연결 근거</p>
          <ul className="mt-1.5 list-disc space-y-1 pl-4 text-[11px] leading-5 text-slate">
            {evidence.map((item) => (
              <li key={item.key}>{item.label}</li>
            ))}
          </ul>
        </div>
      )}

      <p
        id={descriptionId}
        className="mt-3 border-t border-[#dde3ea] pt-2.5 text-[10px] leading-4 text-slate"
      >
        검색 유사도 점수는 동일 민원 판정이나 발생 가능성을 뜻하지 않습니다.
      </p>
    </article>
  );
}

function summaryValue(summary, keys) {
  return firstPresentValue(summary, keys);
}

export function ComplaintReferenceSummary({ summary }) {
  if (!summary) return null;

  const matched = summaryValue(summary, [
    "matched",
    "matched_count",
    "matched_complaints",
    "matched_with_reference_case",
  ]);
  const total = summaryValue(summary, [
    "total",
    "total_count",
    "total_complaints",
    "searched_complaints",
  ]);
  const unavailableCount = summaryValue(summary, [
    "unavailable",
    "unavailable_count",
    "search_unavailable_count",
  ]);
  const invalidCount = summaryValue(summary, ["invalid", "invalid_count"]);
  const status = summary.status ?? summary.search_status;
  const isUnavailable = status === "unavailable";
  const hasOnlyInvalidQueries =
    typeof total === "number" &&
    total > 0 &&
    typeof invalidCount === "number" &&
    invalidCount === total;
  const isPartial =
    status === "partial" ||
    status === "partially_unavailable" ||
    (typeof unavailableCount === "number" && unavailableCount > 0);

  let value = "검색 불가";
  if (hasOnlyInvalidQueries) {
    value = "검색 대상 없음";
  } else if (isUnavailable) {
    value = "검색 불가";
  } else if (isPartial) {
    value = "일부 검색";
  } else if (matched !== null && total !== null) {
    value = `${matched}/${total}건`;
  }
  const detail = hasOnlyInvalidQueries
    ? "민원 요약이 부족해 참고 사례를 검색하지 않음"
    : isPartial && matched !== null && total !== null
      ? `${matched}/${total}건 연결 · 조건 검증을 통과한 공개 Q&A 사례`
      : "조건 검증을 통과한 공개 Q&A 사례";

  return (
    <div className="rounded-[14px] bg-[#f0f4f9] px-4 py-3 text-right">
      <p className="text-[11px] font-medium text-slate">참고사례 연결</p>
      <p className="mt-0.5 text-[20px] font-bold text-ink">{value}</p>
      <p className="max-w-[180px] text-[10px] leading-4 text-slate">
        {detail}
      </p>
    </div>
  );
}
