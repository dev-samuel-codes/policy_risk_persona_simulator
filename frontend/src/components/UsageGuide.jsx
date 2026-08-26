const STEPS = [
  {
    n: "1",
    title: "문서 입력",
    text: "정책 화면에서 문서를 직접 입력하거나 파일로 업로드합니다.",
  },
  {
    n: "2",
    title: "시뮬레이션 실행",
    text: "인구 통계를 반영한 시민 페르소나가 자동으로 생성되어 조항별 반응을 만듭니다.",
  },
  {
    n: "3",
    title: "결과 확인",
    text: "페르소나별 적용·제외 사유와 예상 민원, 조건이 확인된 공개 Q&A 참고 사례를 확인합니다.",
  },
];

export default function UsageGuide() {
  return (
    <div>
      <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-slate">
        사용법
      </p>

      <ol className="mt-5 flex flex-col divide-y divide-line border-y border-line">
        {STEPS.map((step) => (
          <li key={step.n} className="flex items-start gap-3.5 py-4">
            <span className="mt-0.5 flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-[6px] bg-brand text-[11px] font-bold text-white">
              {step.n}
            </span>
            <div className="min-w-0">
              <p className="text-[14px] font-semibold text-ink">{step.title}</p>
              <p className="mt-1 text-[13px] leading-6 text-slate">{step.text}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
