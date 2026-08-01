// TODO: 목업 페르소나 3명. 추후 Persona Cohort Sampler 결과로 교체.
const SAMPLE_PERSONAS = [
  { id: "p1", name: "김복실", age: 67, job: "무직", personality: "신중하고 걱정이 많음", note: "고령 1인가구", photo: null },
  { id: "p2", name: "박영남", age: 44, job: "일용직 근로자", personality: "직설적이고 실용적", note: "저소득 가구주", photo: null },
  { id: "p3", name: "최준구", age: 30, job: "계약직 사무원", personality: "정보 탐색에 적극적", note: "청년 1인가구", photo: null },
];

function Avatar({ name, photo }) {
  return (
    <div className="flex h-[76px] w-[76px] items-center justify-center overflow-hidden rounded-full bg-brand-soft">
      {photo ? (
        <img src={photo} alt={name} className="h-full w-full object-cover" />
      ) : (
        <svg viewBox="0 0 24 24" className="h-9 w-9 text-brand/70" fill="none" aria-hidden="true">
          <circle cx="12" cy="8" r="4" fill="currentColor" />
          <path
            d="M4 20c0-4.4 3.6-7 8-7s8 2.6 8 7"
            fill="currentColor"
          />
        </svg>
      )}
    </div>
  );
}

function PersonaTooltip({ p }) {
  return (
    <div className="absolute left-[calc(100%+12px)] top-0 z-50 w-[200px] rounded-card border border-line bg-white p-4 text-left shadow-lg">
      <p className="text-[14px] font-semibold text-ink">{p.name}</p>
      <dl className="mt-2 flex flex-col gap-1 text-[12px] text-slate">
        <div className="flex gap-1.5">
          <dt className="shrink-0 text-ink/60">나이</dt>
          <dd>{p.age}세</dd>
        </div>
        <div className="flex gap-1.5">
          <dt className="shrink-0 text-ink/60">직업</dt>
          <dd>{p.job}</dd>
        </div>
        <div className="flex gap-1.5">
          <dt className="shrink-0 text-ink/60">성격</dt>
          <dd>{p.personality}</dd>
        </div>
        <div className="flex gap-1.5">
          <dt className="shrink-0 text-ink/60">특징</dt>
          <dd>{p.note}</dd>
        </div>
      </dl>
    </div>
  );
}

export default function PersonaSidebar() {
  return (
    <aside className="flex w-[164px] shrink-0 flex-col gap-6 border-r border-line px-4 py-8">
      <button
        type="button"
        className="flex items-center justify-center gap-1.5 rounded-pill border border-dashed border-line py-2.5 text-[13px] font-medium text-slate transition-colors hover:border-brand hover:text-brand"
      >
        <span className="text-[16px] leading-none">+</span> 새로 생성
      </button>

      <ul className="flex flex-col gap-6">
        {SAMPLE_PERSONAS.map((p) => (
          <li key={p.id} className="group relative">
            <button type="button" className="flex w-full flex-col items-center gap-2">
              <Avatar name={p.name} />
              <span className="text-[14px] font-medium text-ink">{p.name}</span>
              <span className="text-[11px] text-slate">{p.note}</span>
            </button>

            <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100">
              <PersonaTooltip p={p} />
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
}