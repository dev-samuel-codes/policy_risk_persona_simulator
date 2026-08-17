import { useRef, useState } from "react";
import { MessageCircle, PencilLine, Upload } from "lucide-react";
import ChatEmptyState from "../components/ChatEmptyState";
import SimulationLayout from "../components/SimulationLayout";
import policyHero from "../assets/images/policy-hero-figma.png";
import policyHeroCircle from "../assets/images/policy-circle-white.svg";

const ACCEPTED_POLICY_FILES = ".pdf,.docx,.hwp,.hwpx,.txt,.md";
const SIMULATION_POLL_INTERVAL_MS = 3000;
const SIMULATION_MAX_POLLS = 600;
const INPUT_CLASS =
  "mt-2 w-full rounded-[12px] border border-[#d8dde5] bg-white px-4 py-3.5 text-[15px] leading-6 text-ink placeholder:text-slate/55 focus:border-brand focus:outline-none";

function PolicyField({
  label,
  name,
  placeholder,
  required = false,
  rows,
  type = "text",
  defaultValue,
}) {
  return (
    <label className="block text-[15px] font-semibold text-ink">
      {label}
      {required && <span className="ml-1 text-brand">*</span>}
      {rows ? (
        <textarea
          name={name}
          rows={rows}
          required={required}
          placeholder={placeholder}
          defaultValue={defaultValue}
          className={`${INPUT_CLASS} resize-y`}
        />
      ) : (
        <input
          name={name}
          type={type}
          required={required}
          placeholder={placeholder}
          defaultValue={defaultValue}
          className={INPUT_CLASS}
        />
      )}
    </label>
  );
}

function DirectPolicyForm({
  onSimulationStarted,
  initialValues,
  heading = "정책 직접 입력",
  subheading,
}) {
  const values = initialValues ?? {};
  const [canSubmit, setCanSubmit] = useState(
    Boolean(values.policy_name?.trim() && values.benefits?.trim()),
  );
  const [submissionState, setSubmissionState] = useState("idle");
  const [submissionError, setSubmissionError] = useState("");
  const isSubmitting = submissionState === "submitting";

  const updateRequiredState = (form) => {
    const policyName = form.elements.namedItem("policy_name")?.value.trim();
    const benefits = form.elements.namedItem("benefits")?.value.trim();
    setCanSubmit(Boolean(policyName && benefits));
    setSubmissionState("idle");
    setSubmissionError("");
  };

  const submitPolicy = async (event) => {
    event.preventDefault();
    if (!canSubmit || isSubmitting) return;

    setSubmissionState("submitting");
    setSubmissionError("");

    try {
      const fields = Object.fromEntries(new FormData(event.currentTarget));
      const response = await fetch("/api/policies/direct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fields),
      });
      const result = await response.json().catch(() => null);

      if (!response.ok) {
        const detail = typeof result?.detail === "string" ? result.detail : null;
        throw new Error(detail ?? "정책 정보를 적용하지 못했습니다.");
      }

      if (!result?.job_id || !result?.policy) {
        throw new Error("시뮬레이션 작업 ID를 받지 못했습니다.");
      }

      onSimulationStarted(result);
    } catch (error) {
      setSubmissionState("error");
      setSubmissionError(
        error instanceof Error ? error.message : "정책 정보를 적용하지 못했습니다.",
      );
    }
  };

  return (
    <main className="min-h-[calc(100vh-76px)] bg-[#f6f7fb] px-8 py-12">
      <form
        className="mx-auto max-w-[1080px] rounded-[24px] border border-[#dde1e8] bg-white p-9 shadow-[0_14px_36px_rgba(20,23,28,0.05)]"
        onChange={(event) => updateRequiredState(event.currentTarget)}
        onSubmit={submitPolicy}
      >
        <h1 className="text-[30px] font-bold tracking-[-0.025em] text-ink">
          {heading}
        </h1>
        {subheading && (
          <p className="mt-2 text-[14px] leading-6 text-slate">{subheading}</p>
        )}

        <fieldset
          disabled={isSubmitting}
          className="mt-8 grid grid-cols-2 gap-x-6 gap-y-6"
        >
          <div className="col-span-2">
            <PolicyField
              label="정책명"
              name="policy_name"
              placeholder="정책명을 입력하세요."
              defaultValue={values.policy_name}
              required
            />
          </div>

          <div className="col-span-2">
            <PolicyField
              label="지원대상"
              name="target_audience"
              placeholder="지원 대상과 자격 요건을 입력하세요."
              defaultValue={values.target_audience}
              rows={3}
            />
          </div>

          <PolicyField
            label="신청기간"
            name="application_period"
            placeholder="예: 2026.08.01 ~ 2026.08.31"
            defaultValue={values.application_period}
          />
          <PolicyField
            label="시행일"
            name="effective_date"
            type="date"
            defaultValue={values.effective_date}
          />

          <PolicyField
            label="제출서류"
            name="required_documents"
            placeholder="필요한 제출서류를 입력하세요."
            defaultValue={values.required_documents}
            rows={4}
          />
          <PolicyField
            label="신청방법"
            name="application_method"
            placeholder="온라인, 방문 등 신청방법을 입력하세요."
            defaultValue={values.application_method}
            rows={4}
          />

          <div className="col-span-2">
            <PolicyField
              label="문의처"
              name="contact"
              placeholder="담당 기관, 부서, 전화번호 등을 입력하세요."
              defaultValue={values.contact}
            />
          </div>

          <PolicyField
            label="지원금(혜택)"
            name="benefits"
            placeholder="지원 금액과 제공 혜택을 입력하세요."
            defaultValue={values.benefits}
            required
            rows={4}
          />
          <PolicyField
            label="제외조건"
            name="exclusion_conditions"
            placeholder="지원에서 제외되는 조건을 입력하세요."
            defaultValue={values.exclusion_conditions}
            rows={4}
          />
        </fieldset>

        <div className="mt-8 flex items-center justify-end gap-5 border-t border-[#e5e8ed] pt-7">
          {submissionState === "error" && (
            <p role="alert" className="text-[14px] font-medium text-[#9a342b]">
              {submissionError}
            </p>
          )}
          <button
            type="submit"
            disabled={!canSubmit || isSubmitting}
            className={`min-w-[160px] rounded-[13px] px-7 py-3.5 text-[16px] font-semibold text-white transition ${
              canSubmit && !isSubmitting
                ? "bg-brand shadow-[0_8px_18px_rgba(44,74,110,0.18)] hover:bg-brand-strong hover:shadow-[0_10px_22px_rgba(44,74,110,0.23)]"
                : "cursor-not-allowed bg-[#b8c1cc]"
            }`}
          >
            {isSubmitting ? "적용 중..." : "입력 완료"}
          </button>
        </div>
      </form>
    </main>
  );
}

function buildCitizenProfile(result, index) {
  const summary = result.persona_summary ?? {};
  const persona = result.persona ?? {};
  const province = persona.province ?? "";
  const district = persona.district ?? "";
  const location =
    summary["거주지"] ?? [province, district].filter(Boolean).join("-");

  return {
    id: result.persona_id ?? persona.uuid ?? `citizen-${index + 1}`,
    name: summary["이름"] ?? `시민 ${index + 1}`,
    age: summary["나이"] ?? persona.age,
    job: summary["직업"] ?? persona.occupation ?? "직업 정보 없음",
    sex: summary["성별"] ?? persona.sex,
    note: location || "거주지 정보 없음",
    personality: result.personality ?? persona.persona,
    photo: null,
    result,
  };
}

function PolicySummaryCard({ policy }) {
  const detail = policy?.상세정보 ?? {};
  const list = policy?.목록정보 ?? {};
  const policyName = detail.서비스명 ?? list.서비스명 ?? "입력한 정책";
  const policyDetails = [
    ["지원 대상", detail.지원대상 ?? list.지원대상],
    ["지원 내용", detail.지원내용 ?? list.지원내용],
    ["신청 기간", detail.신청기한 ?? list.신청기한],
    ["시행일", detail.시행일],
    ["제출 서류", detail.구비서류],
    ["신청 방법", detail.신청방법 ?? list.신청방법],
    ["제외 조건", detail.제외조건 ?? detail.선정기준],
    ["문의처", detail.문의처 ?? list.전화문의],
  ].filter(([, value]) => typeof value === "string" && value.trim());

  return (
    <article className="max-h-[680px] overflow-y-auto rounded-[22px] border border-line bg-white p-6 shadow-[0_12px_30px_rgba(20,23,28,0.05)]">
      <p className="text-[13px] font-semibold text-brand">입력된 정책</p>
      <h2 className="mt-2 text-[22px] font-bold leading-snug tracking-[-0.02em] text-ink">
        {policyName}
      </h2>

      {policyDetails.length > 0 && (
        <dl className="mt-6 divide-y divide-line border-y border-line">
          {policyDetails.map(([label, value]) => (
            <div key={label} className="py-4">
              <dt className="text-[12px] font-semibold text-slate">{label}</dt>
              <dd className="mt-1.5 whitespace-pre-line text-[13px] leading-6 text-ink">
                {value}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </article>
  );
}

function SimilarPoliciesPanel({ policies = [], similarity }) {
  const hasPolicies = policies.length > 0;

  return (
    <section className="mx-auto w-full max-w-[1120px] rounded-[22px] border border-line bg-white p-6 shadow-[0_12px_30px_rgba(20,23,28,0.05)]">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-[13px] font-semibold text-brand">
            기존 정책 데이터 기반
          </p>
          <h2 className="mt-1 text-[24px] font-bold tracking-[-0.025em] text-ink">
            유사한 과거 정책
          </h2>
        </div>
        {similarity?.as_of_date && (
          <p className="text-[12px] text-slate">
            {similarity.as_of_date} 이전 등록 정책 · {similarity.source_count?.toLocaleString()}건 검색
          </p>
        )}
      </div>

      {!hasPolicies ? (
        <p className="mt-5 rounded-[14px] bg-surface px-4 py-4 text-[14px] text-slate">
          기준 이상의 유사 정책을 찾지 못했습니다.
        </p>
      ) : (
        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          {policies.map((policy, index) => (
            <article
              key={policy.service_id}
              className="rounded-[18px] border border-[#dde3ea] bg-[#fbfcfd] p-5"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="text-[12px] font-semibold text-brand">
                    유사 정책 {index + 1}
                  </p>
                  <h3 className="mt-1 text-[18px] font-bold leading-7 text-ink">
                    {policy.policy_name}
                  </h3>
                  <p className="mt-1 text-[12px] text-slate">
                    {[policy.organization, policy.registered_at]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                </div>
                <span className="shrink-0 rounded-pill bg-brand-soft px-3 py-1.5 text-[13px] font-bold text-brand">
                  {policy.similarity_score}%
                </span>
              </div>

              {policy.match_reasons?.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {policy.match_reasons.map((reason) => (
                    <span
                      key={`${policy.service_id}-${reason.field}`}
                      className="rounded-pill bg-white px-2.5 py-1 text-[11px] font-medium text-slate ring-1 ring-line"
                    >
                      {reason.field} {reason.score}% 일치
                    </span>
                  ))}
                </div>
              )}

              {policy.target_audience && (
                <div className="mt-4">
                  <p className="text-[11px] font-semibold text-slate">지원 대상</p>
                  <p className="mt-1 max-h-20 overflow-hidden text-[13px] leading-5 text-ink">
                    {policy.target_audience}
                  </p>
                </div>
              )}
              {policy.benefits && (
                <div className="mt-3">
                  <p className="text-[11px] font-semibold text-slate">지원 내용</p>
                  <p className="mt-1 max-h-20 overflow-hidden text-[13px] leading-5 text-ink">
                    {policy.benefits}
                  </p>
                </div>
              )}

              {policy.source_url && (
                <a
                  href={policy.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-4 inline-flex text-[12px] font-semibold text-brand hover:underline"
                >
                  정책 원문 보기
                </a>
              )}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function PersonaDialogueView({ policy, profile, riskScore }) {
  const complaints = profile.result?.complaints ?? [];
  const profileSummary = [
    profile.age ? `${profile.age}세` : null,
    profile.job,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <section className="mx-auto w-full max-w-[1120px] pb-4">
      <div className="flex items-start justify-between gap-6">
        <div>
          <p className="text-[13px] font-semibold text-brand">시민 시뮬레이션 결과</p>
          <h1 className="mt-1 text-[28px] font-bold tracking-[-0.025em] text-ink">
            시민 반응
          </h1>
        </div>
        {typeof riskScore?.score === "number" && (
          <div className="rounded-[14px] bg-brand-soft px-4 py-3 text-right">
            <p className="text-[11px] font-medium text-slate">정책 민원 리스크</p>
            <p className="mt-0.5 text-[20px] font-bold text-brand">
              {riskScore.score}점
            </p>
          </div>
        )}
      </div>

      <div className="mt-6 grid items-start gap-6 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <PolicySummaryCard policy={policy} />

        <article className="rounded-[22px] border border-line bg-white p-7 shadow-[0_12px_30px_rgba(20,23,28,0.05)]">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <h2 className="text-[21px] font-bold text-ink">{profile.name}</h2>
            <span className="rounded-pill bg-surface px-3 py-1 text-[12px] text-slate">
              {profileSummary}
            </span>
            <span className="rounded-pill bg-surface px-3 py-1 text-[12px] text-slate">
              {profile.note}
            </span>
          </div>

          {profile.personality && (
            <p className="mt-4 rounded-[14px] bg-[#f7f8fa] px-4 py-3 text-[14px] leading-6 text-slate">
              {profile.personality}
            </p>
          )}

          <div className="mt-6 flex flex-col gap-4">
            {complaints.map((complaint, index) => (
              <div
                key={`${profile.id}-complaint-${index + 1}`}
                className="rounded-[18px] border border-[#dde3ea] bg-white p-5"
              >
                <div className="flex items-center gap-2 text-[13px] font-semibold text-brand">
                  <MessageCircle aria-hidden="true" size={17} strokeWidth={1.8} />
                  민원 대사 {index + 1}
                </div>
                <blockquote className="mt-3 text-[17px] font-medium leading-8 text-ink">
                  “{complaint.dialogue}”
                </blockquote>
                {complaint.complaint_text && (
                  <p className="mt-3 border-t border-line pt-3 text-[13px] leading-5 text-slate">
                    {complaint.complaint_text}
                  </p>
                )}
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  );
}

function SimulationRunScreen({ job }) {
  const [selectedPersonaId, setSelectedPersonaId] = useState(null);
  const policyName =
    job.policy?.상세정보?.서비스명 ?? job.policy?.목록정보?.서비스명 ?? "입력한 정책";
  const citizenResults = job.result?.citizen_results ?? [];
  const profiles = citizenResults.map(buildCitizenProfile);
  const activePersonaId = profiles.some(
    (profile) => profile.id === selectedPersonaId,
  )
    ? selectedPersonaId
    : profiles[0]?.id;
  const activeProfile = profiles.find(
    (profile) => profile.id === activePersonaId,
  );

  let title = null;
  if (job.status === "completed" && profiles.length === 0) {
    title = "시뮬레이션은 완료되었지만 생성된 시민 대사가 없습니다.";
  } else if (job.status === "failed") {
    title = job.error || "시뮬레이션 실행에 실패했습니다.";
  }

  return (
    <SimulationLayout
      placeholder="분석할 정책안을 입력하시오."
      personas={profiles.length > 0 ? profiles : undefined}
      selectedPersonaId={activePersonaId}
      onSelectPersona={setSelectedPersonaId}
    >
      <div className="flex flex-col gap-8">
        <SimilarPoliciesPanel
          policies={job.similar_policies}
          similarity={job.similarity}
        />
        {job.status === "completed" && activeProfile ? (
          <PersonaDialogueView
            policy={job.policy}
            profile={activeProfile}
            riskScore={job.result?.risk_score}
          />
        ) : (
          title && <ChatEmptyState title={title} />
        )}
      </div>
    </SimulationLayout>
  );
}

export default function PolicyPage() {
  const fileInputRef = useRef(null);
  const [inputMode, setInputMode] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [simulationJob, setSimulationJob] = useState(null);
  const [extractionStatus, setExtractionStatus] = useState("idle");
  const [extractionError, setExtractionError] = useState("");
  const [extractedFields, setExtractedFields] = useState(null);
  const isExtracting = extractionStatus === "extracting";

  const monitorSimulation = async (jobId) => {
    try {
      for (let pollCount = 0; pollCount < SIMULATION_MAX_POLLS; pollCount += 1) {
        const response = await fetch(`/api/simulations/${jobId}`);
        const job = await response.json().catch(() => null);

        if (!response.ok) {
          const detail = typeof job?.detail === "string" ? job.detail : null;
          throw new Error(detail ?? "시뮬레이션 상태를 확인하지 못했습니다.");
        }

        setSimulationJob((currentJob) =>
          currentJob?.job_id === jobId ? job : currentJob,
        );

        if (job.status === "completed" || job.status === "failed") return;

        await new Promise((resolve) =>
          window.setTimeout(resolve, SIMULATION_POLL_INTERVAL_MS),
        );
      }

      throw new Error("시뮬레이션 실행 시간이 초과되었습니다.");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "시뮬레이션 실행에 실패했습니다.";
      setSimulationJob((currentJob) =>
        currentJob?.job_id === jobId
          ? { ...currentJob, status: "failed", error: message }
          : currentJob,
      );
    }
  };

  const openSimulationScreen = (job) => {
    setSimulationJob({
      job_id: job.job_id,
      status: "queued",
      policy: job.policy,
      similar_policies: job.similar_policies ?? [],
      similarity: job.similarity ?? null,
      result: null,
      error: null,
    });
    void monitorSimulation(job.job_id);
  };

  const selectFile = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setInputMode("file");
    setExtractionStatus("idle");
    setExtractionError("");
    setExtractedFields(null);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  };

  const extractSelectedFile = async () => {
    if (!selectedFile || isExtracting) return;

    setExtractionStatus("extracting");
    setExtractionError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("/api/policies/extract-file", {
        method: "POST",
        body: formData,
      });
      const result = await response.json().catch(() => null);

      if (!response.ok) {
        const detail = typeof result?.detail === "string" ? result.detail : null;
        throw new Error(detail ?? "파일에서 정책 정보를 추출하지 못했습니다.");
      }

      if (!result?.fields) {
        throw new Error("추출된 정책 정보를 받지 못했습니다.");
      }

      setExtractedFields(result.fields);
      setExtractionStatus("idle");
      setInputMode("file-review");
    } catch (error) {
      setExtractionStatus("error");
      setExtractionError(
        error instanceof Error
          ? error.message
          : "파일에서 정책 정보를 추출하지 못했습니다.",
      );
    }
  };

  if (simulationJob) {
    return <SimulationRunScreen job={simulationJob} />;
  }

  if (inputMode === "text") {
    return <DirectPolicyForm onSimulationStarted={openSimulationScreen} />;
  }

  if (inputMode === "file-review" && extractedFields) {
    return (
      <DirectPolicyForm
        heading="파일에서 추출한 정책 정보"
        subheading="자동으로 추출한 내용입니다. 실행 전에 내용을 확인하고 필요한 부분을 수정하세요."
        initialValues={extractedFields}
        onSimulationStarted={openSimulationScreen}
      />
    );
  }

  return (
    <main className="min-h-[calc(100vh-76px)] bg-[#f6f7fb] lg:h-[calc(100vh-76px)] lg:overflow-hidden">
      <section className="relative mx-auto grid max-w-[1440px] grid-cols-1 lg:h-full lg:grid-cols-[270px_1fr]">
        <div className="hidden bg-[#f6f7fb] lg:block" />

        <div className="relative z-10 flex flex-col justify-center bg-white px-6 py-10 sm:px-12 lg:h-full lg:py-10 lg:pl-[330px] lg:pr-16">
          <h1 className="text-[26px] font-bold leading-tight tracking-[-0.03em] text-ink sm:text-[32px] lg:text-[36px]">
            정책 분석
          </h1>
          <p className="mt-2 text-[13px] leading-6 text-slate sm:text-[14px] lg:mt-3 lg:max-w-[460px] lg:text-[15px]">
            분석할 정책 파일을 업로드하거나 정책 내용을 직접 입력하세요.
          </p>

          <div
            className={`mt-6 w-full max-w-[620px] rounded-[20px] border px-4 py-5 transition-colors sm:px-6 sm:py-6 ${
              isDragging
                ? "border-brand bg-brand-soft"
                : "border-[#dde1e8] bg-white"
            }`}
            onDragEnter={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <div className="grid gap-2.5 sm:grid-cols-2 sm:gap-3">
              <button
                type="button"
                className="flex min-h-[56px] items-center justify-center gap-2 rounded-[13px] bg-brand px-5 text-[14px] font-semibold text-white shadow-[0_8px_18px_rgba(44,74,110,0.18)] transition hover:bg-brand-strong hover:shadow-[0_10px_24px_rgba(44,74,110,0.23)]"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload aria-hidden="true" size={19} strokeWidth={1.8} />
                정책 파일 업로드
              </button>

              <button
                type="button"
                className="flex min-h-[56px] items-center justify-center gap-2 rounded-[13px] border border-[#cfd6df] bg-white px-5 text-[14px] font-semibold text-brand transition hover:border-brand hover:bg-brand-soft/60"
                onClick={() => setInputMode("text")}
              >
                <PencilLine aria-hidden="true" size={19} strokeWidth={1.8} />
                정책 직접 입력
              </button>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_POLICY_FILES}
              className="sr-only"
              onChange={(event) => selectFile(event.target.files?.[0])}
            />

            <p className="mt-3 text-[12px] text-slate">
              PDF, DOCX, HWP, HWPX, TXT 파일을 끌어다 놓아도 됩니다.
            </p>
          </div>

          {inputMode === "file" && selectedFile && (
            <div className="mt-5 flex w-full max-w-[740px] flex-col gap-4 rounded-[18px] border border-[#dde1e8] bg-white px-5 py-4 text-left shadow-[0_10px_24px_rgba(20,23,28,0.05)]">
              <div className="flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <p className="truncate text-[15px] font-semibold text-ink">
                    {selectedFile.name}
                  </p>
                  <p className="mt-1 text-[13px] text-slate">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
                <button
                  type="button"
                  disabled={isExtracting}
                  className="shrink-0 rounded-[10px] px-3 py-2 text-[14px] font-medium text-brand transition hover:bg-brand-soft disabled:cursor-not-allowed disabled:opacity-50"
                  onClick={() => fileInputRef.current?.click()}
                >
                  파일 변경
                </button>
              </div>

              {extractionStatus === "error" && (
                <p role="alert" className="text-[13px] font-medium text-[#9a342b]">
                  {extractionError}
                </p>
              )}

              <button
                type="button"
                disabled={isExtracting}
                onClick={extractSelectedFile}
                className={`flex min-h-[52px] items-center justify-center rounded-[13px] px-6 text-[15px] font-semibold text-white transition ${
                  isExtracting
                    ? "cursor-not-allowed bg-[#b8c1cc]"
                    : "bg-brand shadow-[0_8px_18px_rgba(44,74,110,0.18)] hover:bg-brand-strong hover:shadow-[0_10px_22px_rgba(44,74,110,0.23)]"
                }`}
              >
                {isExtracting ? "파일에서 정책 정보를 추출하는 중..." : "파일에서 정책 정보 추출"}
              </button>
            </div>
          )}
        </div>

        <img
          src={policyHeroCircle}
          alt=""
          aria-hidden="true"
          className="pointer-events-none absolute left-[60px] top-1/2 z-0 hidden h-[480px] w-[480px] -translate-y-1/2 lg:block"
        />
        <div className="pointer-events-none absolute left-[80px] top-1/2 z-20 hidden h-[440px] w-[440px] -translate-y-1/2 overflow-hidden rounded-full shadow-[12px_9px_18px_rgba(15,23,42,0.45)] lg:block">
          <img
            src={policyHero}
            alt=""
            aria-hidden="true"
            className="h-full w-full scale-110 object-cover"
          />
        </div>
      </section>
    </main>
  );
}
