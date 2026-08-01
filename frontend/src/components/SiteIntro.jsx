export default function SiteIntro() {
  return (
    <div className="flex flex-col gap-10 lg:flex-row lg:items-start">
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-medium tracking-wide text-brand">사이트 소개</p>
        <h1 className="mt-2 text-[26px] font-semibold leading-snug text-ink">
          정책이 시행되기 전에,
          <br />
          생길 문제를 먼저 시뮬레이션합니다.
        </h1>
        <p className="mt-4 max-w-[420px] text-[14px] leading-relaxed text-slate">
          CivicEcho는 정책·법령 문서와 인구 통계를 바탕으로 가상의 시민 페르소나를 생성하고,
          각 조항이 서로 다른 계층에 어떻게 적용되는지를 시뮬레이션해 예상 민원을
          리스크 지표로 정리합니다.
        </p>
      </div>

      <div className="w-full min-w-0 max-w-[620px] shrink-0">
        <Pipeline />

        <div className="mt-4 w-full rounded-[14px] bg-white px-5 py-3">
          <p className="whitespace-nowrap text-[13px] leading-relaxed text-ink">
            시민에게는{" "}
            <span className="font-semibold text-brand">&ldquo;나에게 적용되는가&rdquo;</span>를,
            공무원에게는{" "}
            <span className="font-semibold text-[var(--color-accent-teal)]">
              &ldquo;어떤 민원이 왜 생기는가&rdquo;
            </span>
            를 보여줍니다.
          </p>
        </div>
      </div>
    </div>
  );
}

function Pipeline() {
  return (
    <div className="grid grid-cols-[1fr_20px_1fr_20px_1fr] items-center gap-2">
      <PipelineStep label="정책 문서" />
      <Arrow />
      <PipelineStep label="페르소나 시뮬레이션" />
      <Arrow />
      <PipelineStep label="리스크 지표" />
    </div>
  );
}

function PipelineStep({ label }) {
  return (
    <div className="flex h-[76px] items-center justify-center rounded-[16px] bg-white px-2 text-center text-[13px] font-semibold leading-snug text-brand">
      {label}
    </div>
  );
}

function Arrow() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="shrink-0 text-ink/25" aria-hidden="true">
      <path d="M4 12H20M20 12L14 6M20 12L14 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}