import { Navigate, Routes, Route } from "react-router-dom";
import TopNav from "./components/TopNav";
import HomePage from "./pages/HomePage";
import PolicyPage from "./pages/PolicyPage";
import AboutPage from "./pages/AboutPage";

export default function App() {
  return (
    <div className="min-h-screen">
      <TopNav />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/policy" element={<PolicyPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
