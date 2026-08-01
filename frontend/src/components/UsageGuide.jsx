const STEPS = [
  { n: "1", text: "왼쪽 사이드바에서 확인하고 싶은 페르소나를 선택하거나 새로 생성" },
  { n: "2", text: "정책 화면 또는 법령 화면에서 분석할 문서를 입력창에 붙여넣기" },
  { n: "3", text: "페르소나별 적용 결과와 리스크 지수를 확인" },
];

export default function UsageGuide() {
  return (
    <div>
      <p className="mb-4 text-[13px] font-medium tracking-wide text-slate">사용법</p>
      <ol className="flex flex-col gap-4">
        {STEPS.map((s) => (
          <li key={s.n} className="flex items-start gap-3">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--color-accent-sand-soft)] text-[12px] font-semibold text-[var(--color-accent-sand)]">
              {s.n}
            </span>
            <span className="text-[14px] leading-relaxed text-ink">{s.text}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}