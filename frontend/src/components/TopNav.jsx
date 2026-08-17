import { NavLink } from "react-router-dom";
import logo from "../assets/images/logo.svg";

const NAV_ITEMS = [
  { to: "/policy", label: "정책" },
  { to: "/law", label: "법령" },
  { to: "/about", label: "소개" },
];

export default function TopNav() {
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-paper/95 backdrop-blur">
      <div className="mx-auto flex h-[76px] max-w-[1440px] items-center justify-between pl-8 pr-24">
        <NavLink to="/" className="flex items-center gap-3">
          <img src={logo} alt="Vecho" className="h-6 w-auto" />
          <span className="hidden text-[14px] leading-tight text-slate md:block">
            법과 정책의 <b className="font-semibold text-ink">내일</b>을 미리 그리다,
            <br />
            시민과 함께 준비하는 행정
          </span>
        </NavLink>

        <nav className="flex h-full items-center gap-12 self-stretch">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `relative flex h-full items-center text-[18px] font-medium transition-colors ${
                  isActive ? "text-brand" : "text-ink/70 hover:text-ink"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {item.label}
                  {isActive && (
                    <span className="absolute -bottom-[1px] left-0 h-[2px] w-full bg-brand" />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
