import { Link } from "react-router-dom";
import { FileText, ArrowRight } from "lucide-react";

const ENTRIES = [
  {
    to: "/policy",
    Icon: FileText,
    label: "정책 분석",
    detail: "지원사업·복지정책의 지원대상과 제외조건을 시민 관점에서 검증합니다.",
    examples: ["청년 월세 지원", "1인 가구 특별공급", "기초연금 수급"],
    accent: "var(--color-accent-teal)",
    accentSoft: "var(--color-accent-teal-soft)",
  },
];

export default function NoticeBoard() {
  return (
    <div>
      <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-slate">
        바로 시작하기
      </p>

      <div className="mt-5 flex flex-col gap-3">
        {ENTRIES.map(
          ({ to, Icon, label, detail, examples, accent, accentSoft }) => (
            <Link
              key={to}
              to={to}
              style={{ borderLeftColor: accent }}
              className="group rounded-[14px] border border-line border-l-[3px] bg-white px-5 py-4 transition hover:shadow-[0_6px_18px_rgba(20,23,28,0.07)]"
            >
              <div className="flex items-center gap-2.5">
                <span
                  style={{ backgroundColor: accentSoft, color: accent }}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[9px]"
                >
                  <Icon aria-hidden="true" size={17} strokeWidth={1.9} />
                </span>
                <span className="text-[15px] font-bold text-ink">{label}</span>
                <ArrowRight
                  aria-hidden="true"
                  size={15}
                  strokeWidth={2}
                  style={{ color: accent }}
                  className="ml-auto shrink-0 opacity-45 transition group-hover:translate-x-0.5 group-hover:opacity-100"
                />
              </div>

              <p className="mt-2.5 text-[13px] leading-6 text-slate">{detail}</p>

              <div className="mt-3 flex flex-wrap gap-1.5">
                {examples.map((example) => (
                  <span
                    key={example}
                    style={{ backgroundColor: accentSoft, color: accent }}
                    className="rounded-pill px-2.5 py-1 text-[11.5px] font-semibold"
                  >
                    {example}
                  </span>
                ))}
              </div>
            </Link>
          ),
        )}
      </div>

      <p className="mt-4 text-[12px] leading-5 text-slate/80">
        분석 결과는 생성형 모델이 만든 가상의 시민 반응이며, 실제 민원 접수 결과와
        다를 수 있습니다.
      </p>
    </div>
  );
}
