import SiteIntro from "../components/SiteIntro";
import UsageGuide from "../components/UsageGuide";
import NoticeBoard from "../components/NoticeBoard";

export default function HomePage() {
  return (
    <main className="mx-auto max-w-[1440px] px-8 pb-24 pt-10">
      <section className="rounded-[28px] bg-brand-soft p-10">
        <SiteIntro />
      </section>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-[28px] bg-[var(--color-accent-sand-soft)] p-7">
          <UsageGuide />
        </div>
        <div className="rounded-[28px] bg-[var(--color-accent-teal-soft)] p-7">
          <NoticeBoard />
        </div>
      </div>
    </main>
  );
}