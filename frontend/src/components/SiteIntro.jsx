export default function SiteIntro() {
  return (
    <div className="flex flex-col gap-10 lg:flex-row lg:items-start lg:gap-12">
      <div className="min-w-0 flex-1">
        <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-slate">
          사이트 소개
        </p>
        <h1 className="mt-3 text-[26px] font-bold leading-snug tracking-[-0.02em] text-ink lg:text-[28px]">
          정책이 시행되기 전에,
          <br />
          생길 문제를 먼저 시뮬레이션합니다.
        </h1>
        <p className="mt-5 max-w-[440px] text-[14px] leading-[1.75] text-slate">
          CivicEcho는 정책·법령 문서와 인구 통계를 바탕으로 가상의 시민 페르소나를
          생성하고, 각 조항이 서로 다른 계층에 어떻게 적용되는지를 시뮬레이션해
          예상 민원을 리스크 지표로 정리합니다.
        </p>

        <dl className="mt-7 flex flex-col gap-3 border-t border-line pt-6">
          <div className="flex items-baseline gap-3">
            <dt className="w-[52px] shrink-0 text-[12px] font-semibold text-brand">
              시민
            </dt>
            <dd className="text-[13px] leading-6 text-ink">
              이 정책이 나에게 적용되는지, 왜 제외되는지
            </dd>
          </div>
          <div className="flex items-baseline gap-3">
            <dt className="w-[52px] shrink-0 text-[12px] font-semibold text-brand">
              공무원
            </dt>
            <dd className="text-[13px] leading-6 text-ink">
              어떤 조항에서 어떤 민원이 왜 발생하는지
            </dd>
          </div>
        </dl>
      </div>

      <div className="w-full min-w-0 lg:max-w-[440px] lg:shrink-0">
        <Pipeline />
      </div>
    </div>
  );
}

const PIPELINE_STEPS = [
  {
    n: "01",
    label: "정책 문서 입력",
    detail: "조항·지원대상·제외조건을 구조화",
  },
  {
    n: "02",
    label: "페르소나 시뮬레이션",
    detail: "인구 통계 기반 가상 시민의 반응 생성",
  },
  {
    n: "03",
    label: "리스크 지표 산출",
    detail: "조항별 민원 발생 가능성을 수치화",
  },
];

function Pipeline() {
  return (
    <ol className="flex flex-col">
      {PIPELINE_STEPS.map((step, index) => (
        <li key={step.n} className="relative flex gap-4">
          <div className="flex flex-col items-center">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-brand/25 bg-white text-[11px] font-bold text-brand">
              {step.n}
            </span>
            {index < PIPELINE_STEPS.length - 1 && (
              <span aria-hidden="true" className="w-px flex-1 bg-line" />
            )}
          </div>

          <div className={index < PIPELINE_STEPS.length - 1 ? "pb-6" : ""}>
            <p className="text-[14px] font-semibold leading-8 text-ink">
              {step.label}
            </p>
            <p className="mt-0.5 text-[12.5px] leading-5 text-slate">
              {step.detail}
            </p>
          </div>
        </li>
      ))}
    </ol>
  );
}