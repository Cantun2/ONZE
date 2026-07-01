import { NavLink, Route, Routes } from "react-router-dom";
import { FixturesView } from "./views/FixturesView";
import { MatchDetailView } from "./views/MatchDetailView";
import { BracketView } from "./views/BracketView";
import { CalibrationView } from "./views/CalibrationView";

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          ONZE
          <small>score distributions, honestly</small>
        </div>
        <nav className="nav">
          <NavLink to="/" end>
            Fixtures
          </NavLink>
          <NavLink to="/bracket">Bracket</NavLink>
          <NavLink to="/calibration">Calibration</NavLink>
        </nav>
      </header>

      <main className="content">
        <Routes>
          <Route path="/" element={<FixturesView />} />
          <Route path="/match/:fixtureId" element={<MatchDetailView />} />
          <Route path="/bracket" element={<BracketView />} />
          <Route path="/calibration" element={<CalibrationView />} />
          <Route path="*" element={<FixturesView />} />
        </Routes>
      </main>
    </div>
  );
}
