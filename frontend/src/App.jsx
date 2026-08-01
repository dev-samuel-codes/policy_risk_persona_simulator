import { Routes, Route } from "react-router-dom";
import TopNav from "./components/TopNav";
import HomePage from "./pages/HomePage";
import PolicyPage from "./pages/PolicyPage";
import LawPage from "./pages/LawPage";
import AboutPage from "./pages/AboutPage";

export default function App() {
  return (
      <div className="min-h-screen">      <TopNav />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/policy" element={<PolicyPage />} />
        <Route path="/law" element={<LawPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </div>
  );
}
