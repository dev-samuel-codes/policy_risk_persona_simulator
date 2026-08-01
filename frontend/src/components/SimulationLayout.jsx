import PersonaSidebar from "./PersonaSidebar";
import ChatInputBar from "./ChatInputBar";

export default function SimulationLayout({ placeholder, children }) {
  return (
    <div className="mx-auto flex max-w-[1440px]">
      <PersonaSidebar />

      <main className="flex min-w-0 flex-1 flex-col px-10 py-10">
        <div className="flex-1">{children}</div>

        <div className="mt-6">
          <ChatInputBar placeholder={placeholder} />
        </div>
      </main>
    </div>
  );
}