import logo from "../assets/images/logo.svg";

export default function ChatEmptyState({ title }) {
  return (
    <div className="flex h-full min-h-[420px] flex-col items-center justify-center gap-4 text-center">
      <img src={logo} alt="Vecho" className="h-8 w-auto opacity-90" />
      <h2 className="text-[19px] font-semibold text-ink">{title}</h2>
    </div>
  );
}