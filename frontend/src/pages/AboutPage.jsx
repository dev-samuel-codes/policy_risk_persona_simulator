const AUDIENCES = [
  {
    title: "시민에게",
    body: "새로운 정책이나 법령이 내 상황에 실제로 적용되는지, 어떤 혜택을 놓치고 있는지 미리 확인할 수 있습니다.",
  },
  {
    title: "공무원에게",
    body: "정책 시행 전, 어떤 민원이 얼마나 발생할지와 그 원인을 데이터 기반으로 미리 파악할 수 있습니다.",
  },
];

export default function AboutPage() {
  return (
    <main className="mx-auto max-w-[900px] px-8 py-20">
      <p className="text-[13px] font-medium tracking-wide text-brand">소개</p>
      <h1 className="mt-2 text-[34px] font-semibold leading-tight text-ink">
        정책이 시행되기 전에,
        <br />
        생길 문제를 먼저 시뮬레이션합니다.
      </h1>
      <p className="mt-5 max-w-[640px] text-[15px] leading-relaxed text-slate">
        Vecho는 정책·법령 문서와 인구 통계를 바탕으로 가상의 시민 페르소나를 생성하고,
        각 조항이 서로 다른 계층에 어떻게 적용되는지를 시뮬레이션해 예상 민원을
        리스크 지표로 정리합니다.
      </p>

      <div className="mt-14 grid grid-cols-1 gap-6 sm:grid-cols-2">
        {AUDIENCES.map((a) => (
          <div key={a.title} className="rounded-card border border-line bg-white p-7">
            <h2 className="text-[16px] font-semibold text-ink">{a.title}</h2>
            <p className="mt-2 text-[14px] leading-relaxed text-slate">{a.body}</p>
          </div>
        ))}
      </div>

      <div className="mt-14 rounded-card border border-line bg-surface p-7">
        <h2 className="text-[15px] font-semibold text-ink">데이터 출처</h2>
        <ul className="mt-3 flex flex-col gap-1.5 text-[13px] text-slate">
          <li>Nemotron Personas Korea — 합성 페르소나 데이터셋</li>
          <li>KOSIS 국가통계포털 — 인구·가구 통계</li>
          <li>law.go.kr — 법령 원문</li>
          <li>국민권익위원회 민원 빅데이터 — 민원 통계 (개인정보 비식별 처리)</li>
        </ul>
      </div>
    </main>
  );
}
