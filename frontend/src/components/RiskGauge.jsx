/**
 * 시그니처 요소: 리스크 인덱스(0~100)를 3구간 세그먼트 바로 표현.
 * 홈/정책/법령 화면에서 반복적으로 등장해 "이 서비스는 위험도를 측정한다"는
 * 정체성을 시각적으로 각인시키는 역할을 한다.
 */
const ZONES = [
  { key: "safe", label: "괜찮음", max: 40, color: "var(--color-risk-safe)" },
  { key: "caution", label: "보통", max: 70, color: "var(--color-risk-caution)" },
  { key: "danger", label: "위험", max: 100, color: "var(--color-risk-danger)" },
];

function zoneOf(score) {
  return ZONES.find((z) => score <= z.max) ?? ZONES[ZONES.length - 1];
}

export default function RiskGauge({ score, size = "md" }) {
  const zone = zoneOf(score);
  const barHeight = size === "lg" ? "h-2.5" : "h-1.5";

  return (
    <div className="w-full">
      <div className={`relative w-full overflow-hidden rounded-pill bg-line ${barHeight}`}>
        <div className="absolute inset-y-0 left-0 w-[40%] bg-[var(--color-risk-safe)] opacity-30" />
        <div className="absolute inset-y-0 left-[40%] w-[30%] bg-[var(--color-risk-caution)] opacity-30" />
        <div className="absolute inset-y-0 left-[70%] w-[30%] bg-[var(--color-risk-danger)] opacity-30" />
        <div
          className="absolute inset-y-0 left-0 rounded-pill transition-all"
          style={{ width: `${score}%`, background: zone.color }}
        />
      </div>
      <div className="mt-1.5 flex items-center justify-between text-[12px] text-slate">
        <span>0</span>
        <span className="font-medium" style={{ color: zone.color }}>
          {zone.label} · {score}점
        </span>
        <span>100</span>
      </div>
    </div>
  );
}
