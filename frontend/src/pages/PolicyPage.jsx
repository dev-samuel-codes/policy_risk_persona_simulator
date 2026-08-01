import SimulationLayout from "../components/SimulationLayout";
import ChatEmptyState from "../components/ChatEmptyState";

export default function PolicyPage() {
  return (
    <SimulationLayout placeholder="분석할 정책안을 입력하시오.">
      <ChatEmptyState title="어떤 정책을 확인해볼까요?" />
    </SimulationLayout>
  );
}