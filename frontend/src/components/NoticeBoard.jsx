// TODO: 실제로는 정책/법령 크롤링·수집 파이프라인에서 나온 최신 업데이트 목록으로 교체.
const NOTICES = [
  { tag: "정책", title: "청년 월세 지원 개정안 리스크 분석 완료", time: "2시간 전" },
  { tag: "법령", title: "기초연금법 시행령 개정 반영", time: "어제" },
  { tag: "정책", title: "1인 가구 특별공급 사각지대 시뮬레이션", time: "2일 전" },
];

const TAG_STYLE = {
  정책: "bg-brand-soft text-brand",
  법령: "bg-[var(--color-accent-teal-soft)] text-[var(--color-accent-teal)]",
};

export default function NoticeBoard() {
  return (
    <div>
      <p className="mb-4 text-[13px] font-medium tracking-wide text-slate">공지사항</p>
      <ul className="flex flex-col divide-y divide-line">
        {NOTICES.map((item) => (
          <li key={item.title} className="flex items-start gap-3 py-3.5 first:pt-0 last:pb-0">
            <span
              className={`mt-0.5 shrink-0 rounded-pill px-2.5 py-1 text-[12px] font-medium ${TAG_STYLE[item.tag]}`}
            >
              {item.tag}
            </span>
            <div className="min-w-0">
              <p className="truncate text-[14px] text-ink">{item.title}</p>
              <p className="mt-0.5 text-[12px] text-slate">{item.time}</p>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}