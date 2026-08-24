import { useEffect, useState } from "react";
import { UserRound } from "lucide-react";

// 예상 소요 시간(밀리초). 실측 후 이 값만 바꾸면 진행바 속도가 조정됩니다.
const ESTIMATED_DURATION_MS = 240000; // 4분
const TICK_MS = 1000;
const MAX_RUNNING_PROGRESS = 92;

const STEPS = [
  { label: "정책 정보 정리", detail: "입력한 항목을 구조화하고 있습니다", until: 8 },
  { label: "유사 정책·민원 검색", detail: "기존 정책과 민원 사례를 대조합니다", until: 20 },
  { label: "시민 페르소나 선정", detail: "정책 대상에 맞는 인물을 고릅니다", until: 32 },
  { label: "시민 반응 생성", detail: "페르소나별 반응을 만들고 있습니다", until: 85 },
  { label: "민원 리스크 산출", detail: "조항별 위험도를 계산합니다", until: 100 },
];

function formatElapsed(totalMs) {
  const totalSeconds = Math.floor(totalMs / 1000);
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function SkeletonPersonaCard({ delayMs = 0 }) {
  const pulse = { animationDelay: `${delayMs}ms` };

  return (
    <div className="grid min-w-0 grid-cols-[64px_minmax(0,1fr)] items-start gap-x-4">
      <div
        className="flex h-14 w-14 animate-pulse items-center justify-center rounded-full border border-[#d9e2ed] bg-brand-soft sm:h-16 sm:w-16"
        style={pulse}
      >
        <UserRound aria-hidden="true" className="h-7 w-7 text-brand/25" strokeWidth={1.7} />
      </div>

      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <div className="h-6 w-28 animate-pulse rounded-[8px] bg-[#e6eaf0]" style={pulse} />
          <div className="h-6 w-36 animate-pulse rounded-pill bg-surface" style={pulse} />
          <div className="h-6 w-32 animate-pulse rounded-pill bg-surface" style={pulse} />
        </div>

        <div className="mt-3 rounded-[26px] rounded-tl-[10px] border border-[#edf0f4] bg-white px-6 py-5">
          <div className="h-3.5 w-20 animate-pulse rounded bg-brand-soft" style={pulse} />
          <div className="mt-3 flex flex-col gap-2.5">
            <div className="h-4 w-full animate-pulse rounded bg-[#eef1f5]" style={pulse} />
            <div className="h-4 w-[92%] animate-pulse rounded bg-[#eef1f5]" style={pulse} />
            <div className="h-4 w-[68%] animate-pulse rounded bg-[#eef1f5]" style={pulse} />
          </div>
          <div className="mt-4 border-t border-line pt-3">
            <div className="h-3 w-[54%] animate-pulse rounded bg-[#f0f2f6]" style={pulse} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function SimulationLoadingScreen({ status = "running" }) {
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      setElapsedMs(Date.now() - startedAt);
    }, TICK_MS);

    return () => window.clearInterval(timer);
  }, []);

  // P(t) = P_max × (1 − e^(−t/τ)),  τ = T_est / 2
  const tau = ESTIMATED_DURATION_MS / 2;
  const progress = MAX_RUNNING_PROGRESS * (1 - Math.exp(-elapsedMs / tau));
  const progressPercent = Math.round(progress);

  const activeStepIndex = Math.max(
    0,
    STEPS.findIndex((step) => progressPercent < step.until),
  );
  const activeStep = STEPS[activeStepIndex] ?? STEPS[STEPS.length - 1];
  const isQueued = status === "queued";
  const isOvertime = elapsedMs > ESTIMATED_DURATION_MS;

  let statusMessage = "예상 소요 시간은 3~5분입니다. 창을 닫지 말고 기다려 주세요.";
  if (isQueued) {
    statusMessage = "시뮬레이션 작업을 대기열에 등록하고 있습니다.";
  } else if (isOvertime) {
    statusMessage = "예상보다 시간이 걸리고 있습니다. 조금만 더 기다려 주세요.";
  }

  return (
    <section
      className="w-full pb-4"
      role="status"
      aria-busy="true"
      aria-live="polite"
    >
      {/* 상단 헤더 */}
      <div className="flex flex-wrap items-start justify-between gap-6">
        <div className="min-w-0">
          <p className="text-[13px] font-semibold text-brand">시민 시뮬레이션 진행 중</p>
          <h1 className="mt-1 flex items-center gap-3 text-[28px] font-bold tracking-[-0.025em] text-ink">
            {activeStep.label}
            <span className="flex items-center gap-1" aria-hidden="true">
              {[0, 1, 2].map((dot) => (
                <span
                  key={dot}
                  className="h-1.5 w-1.5 animate-pulse rounded-full bg-brand/60"
                  style={{ animationDelay: `${dot * 180}ms` }}
                />
              ))}
            </span>
          </h1>
          <p className="mt-2 text-[14px] leading-6 text-slate">{activeStep.detail}</p>
        </div>

        <div className="rounded-[14px] bg-brand-soft px-4 py-3 text-right">
          <p className="text-[11px] font-medium text-slate">경과 시간</p>
          <p className="mt-0.5 font-mono text-[20px] font-bold tabular-nums text-brand">
            {formatElapsed(elapsedMs)}
          </p>
        </div>
      </div>

      {/* 진행바 */}
      <div className="mt-7 rounded-[20px] border border-line bg-white p-6">
        <div className="flex items-end justify-between gap-4">
          <p className="text-[13px] font-medium text-slate">{statusMessage}</p>
          <p className="shrink-0 text-[15px] font-bold tabular-nums text-brand">
            {progressPercent}%
          </p>
        </div>

        <div
          className="mt-3 h-2.5 w-full overflow-hidden rounded-pill bg-[#eaeef3]"
          role="progressbar"
          aria-valuenow={progressPercent}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="시뮬레이션 진행률"
        >
          <div
            className="h-full rounded-pill bg-brand transition-[width] duration-1000 ease-linear"
            style={{ width: `${progressPercent}%` }}
          />
        </div>

        {/* 단계 목록 */}
        <ol className="mt-6 flex flex-col gap-3 border-t border-line pt-5">
          {STEPS.map((step, index) => {
            const isDone = index < activeStepIndex;
            const isActive = index === activeStepIndex;

            return (
              <li key={step.label} className="flex items-center gap-3">
                <span
                  aria-hidden="true"
                  className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-bold transition ${
                    isDone
                      ? "bg-brand text-white"
                      : isActive
                        ? "bg-brand-soft text-brand ring-2 ring-brand/30"
                        : "bg-[#eef1f5] text-slate/60"
                  }`}
                >
                  {isDone ? "✓" : index + 1}
                </span>
                <span
                  className={`text-[14px] transition ${
                    isActive
                      ? "font-semibold text-ink"
                      : isDone
                        ? "text-slate"
                        : "text-slate/55"
                  }`}
                >
                  {step.label}
                </span>
                {isActive && (
                  <span className="ml-auto shrink-0 rounded-pill bg-brand-soft px-2.5 py-1 text-[11px] font-semibold text-brand">
                    진행 중
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      </div>

      {/* 결과 자리 스켈레톤 */}
      <div className="mt-7 grid items-start gap-7 xl:grid-cols-[minmax(0,2fr)_1px_minmax(340px,1fr)] xl:gap-8">
        <div className="flex min-w-0 flex-col gap-10">
          <SkeletonPersonaCard delayMs={0} />
          <SkeletonPersonaCard delayMs={220} />
          <SkeletonPersonaCard delayMs={440} />
        </div>

        <div aria-hidden="true" className="hidden h-full min-h-[520px] bg-line xl:block" />

        <aside className="min-w-0 border-t border-line pt-7 xl:border-t-0 xl:pt-0">
          <div className="h-[52px] w-full animate-pulse rounded-[16px] border border-brand/15 bg-brand-soft/45" />
          <div className="mt-4 rounded-[20px] border border-line bg-white p-6">
            <div className="h-3.5 w-24 animate-pulse rounded bg-brand-soft" />
            <div className="mt-3 h-6 w-[70%] animate-pulse rounded bg-[#e6eaf0]" />
            <div className="mt-6 flex flex-col gap-4 border-y border-line py-4">
              {[0, 1, 2, 3].map((row) => (
                <div key={row}>
                  <div
                    className="h-3 w-20 animate-pulse rounded bg-[#eef1f5]"
                    style={{ animationDelay: `${row * 150}ms` }}
                  />
                  <div
                    className="mt-2 h-4 w-[85%] animate-pulse rounded bg-[#f2f4f7]"
                    style={{ animationDelay: `${row * 150}ms` }}
                  />
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}