import { useState } from "react";

const MODES = [
  { key: "citizen", label: "일반" },
  { key: "official", label: "공무원" },
];

export default function ChatInputBar({ placeholder = "입력하시오." }) {
  const [mode, setMode] = useState("citizen");

  return (
    <form
      className="flex flex-col gap-3 rounded-card border border-line bg-white px-6 py-5"
      onSubmit={(e) => e.preventDefault()}
    >
      <div className="flex items-center gap-4">
        <input
          type="text"
          placeholder={placeholder}
          className="min-w-0 flex-1 bg-transparent text-[16px] text-ink placeholder:text-slate/70 focus:outline-none"
        />
        <button
          type="submit"
          aria-label="분석 요청 보내기"
          className="flex h-[52px] w-[52px] shrink-0 items-center justify-center rounded-full bg-surface text-ink transition-colors hover:bg-brand hover:text-white"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M3 11.5L20.5 3.5L14 20.5L11 13L3 11.5Z"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinejoin="round"
              fill="none"
            />
          </svg>
        </button>
      </div>

      <div className="flex items-center gap-2">
        {MODES.map((m) => (
          <button
            key={m.key}
            type="button"
            onClick={() => setMode(m.key)}
            className={`rounded-pill px-4 py-1.5 text-[13px] font-medium transition-colors ${
              mode === m.key
                ? "bg-brand-soft text-brand"
                : "text-slate hover:bg-surface"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>
    </form>
  );
}