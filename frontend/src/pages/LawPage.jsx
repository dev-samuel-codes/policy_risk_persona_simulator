import SimulationLayout from "../components/SimulationLayout";
import ChatEmptyState from "../components/ChatEmptyState";

export default function LawPage() {
  return (
    <SimulationLayout placeholder="분석할 법령 조문을 입력하시오.">
      <ChatEmptyState title="어떤 법령을 확인해볼까요?" />
    </SimulationLayout>
  );
}
