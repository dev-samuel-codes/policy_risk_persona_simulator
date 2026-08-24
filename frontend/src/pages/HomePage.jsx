import SiteIntro from "../components/SiteIntro";
import UsageGuide from "../components/UsageGuide";
import NoticeBoard from "../components/NoticeBoard";

export default function HomePage() {
  return (
    <main className="mx-auto max-w-[1280px] px-6 pb-24 pt-10 sm:px-8">
      <section className="rounded-[20px] border border-line bg-white p-8 sm:p-10">
        <SiteIntro />
      </section>

      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-[1fr_1fr]">
        <section className="rounded-[20px] border border-line bg-surface p-7">
          <UsageGuide />
        </section>
        <section className="rounded-[20px] border border-line bg-white p-7">
          <NoticeBoard />
        </section>
      </div>
    </main>
  );
}