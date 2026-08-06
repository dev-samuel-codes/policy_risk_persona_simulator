import PersonaSidebar from "./PersonaSidebar";

export default function SimulationLayout({
  personas,
  selectedPersonaId,
  onSelectPersona,
  children,
}) {
  return (
    <div className="mx-auto flex max-w-[1440px]">
      <PersonaSidebar
        personas={personas}
        selectedPersonaId={selectedPersonaId}
        onSelectPersona={onSelectPersona}
      />

      <main className="flex min-w-0 flex-1 flex-col px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
        <div className="flex-1">{children}</div>
      </main>
    </div>
  );
}
