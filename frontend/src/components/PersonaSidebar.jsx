import { Plus, UserRound } from "lucide-react";

const SAMPLE_PERSONAS = [
  {
    id: "p1",
    name: "김복실",
    age: 67,
    job: "무직",
    personality: "신중하고 걱정이 많음",
    note: "고령 1인가구",
    photo: null,
  },
  {
    id: "p2",
    name: "박영남",
    age: 44,
    job: "일용직 근로자",
    personality: "직설적이고 실용적",
    note: "저소득 가구주",
    photo: null,
  },
  {
    id: "p3",
    name: "최준구",
    age: 30,
    job: "계약직 사무원",
    personality: "정보 탐색에 적극적",
    note: "청년 1인가구",
    photo: null,
  },
];

function Avatar({ name, photo }) {
  return (
    <div className="flex h-[76px] w-[76px] items-center justify-center overflow-hidden rounded-full bg-brand-soft">
      {photo ? (
        <img src={photo} alt={name} className="h-full w-full object-cover" />
      ) : (
        <UserRound
          aria-hidden="true"
          className="h-9 w-9 text-brand/70"
          strokeWidth={1.8}
        />
      )}
    </div>
  );
}

function ProfileRow({ label, value }) {
  if (!value) return null;

  return (
    <div className="flex gap-2">
      <dt className="shrink-0 text-ink/60">{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function PersonaTooltip({ persona }) {
  return (
    <div className="absolute left-[calc(100%+12px)] top-0 z-50 w-[270px] rounded-card border border-line bg-white p-4 text-left shadow-lg">
      <p className="text-[14px] font-semibold text-ink">{persona.name}</p>
      <dl className="mt-2 flex flex-col gap-1.5 text-[12px] leading-5 text-slate">
        <ProfileRow label="나이" value={persona.age ? `${persona.age}세` : null} />
        <ProfileRow label="직업" value={persona.job} />
        <ProfileRow label="성별" value={persona.sex} />
        <ProfileRow label="거주지" value={persona.note} />
        <ProfileRow label="성격" value={persona.personality} />
      </dl>
    </div>
  );
}

export default function PersonaSidebar({
  personas = SAMPLE_PERSONAS,
  selectedPersonaId,
  onSelectPersona,
}) {
  return (
    <aside className="hidden w-[180px] shrink-0 flex-col gap-6 border-r border-line px-4 py-8 md:flex">
      <button
        type="button"
        className="flex items-center justify-center gap-1.5 rounded-pill border border-dashed border-line py-2.5 text-[13px] font-medium text-slate transition-colors hover:border-brand hover:text-brand"
      >
        <Plus aria-hidden="true" size={16} strokeWidth={1.8} />
        새로 생성
      </button>

      <ul className="flex flex-col gap-5">
        {personas.map((persona) => {
          const isSelected = persona.id === selectedPersonaId;

          return (
            <li key={persona.id} className="group relative">
              <button
                type="button"
                aria-pressed={isSelected}
                className={`flex w-full flex-col items-center gap-2 rounded-[16px] px-2 py-3 transition ${
                  isSelected
                    ? "bg-brand-soft ring-1 ring-brand/20"
                    : "hover:bg-surface"
                }`}
                onClick={() => onSelectPersona?.(persona.id)}
              >
                <Avatar name={persona.name} photo={persona.photo} />
                <span className="text-[14px] font-semibold text-ink">
                  {persona.name}
                </span>
                <span className="max-w-full truncate text-[11px] text-slate">
                  {persona.note || persona.job}
                </span>
              </button>

              <div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100">
                <PersonaTooltip persona={persona} />
              </div>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
