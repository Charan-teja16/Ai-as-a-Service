import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

const cards = [
  {
    key: "csv",
    title: "CSV (Tabular)",
    subtitle: "One .csv (≤300 MB). We auto-detect the target.",
    bullets: [],
  },
  {
    key: "supervised",
    title: "Images with Labelled Dataset",
    subtitle: ".zip with one folder per class. We train a CNN.",
    bullets: [],
  },
];

export default function ModeSelect() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!loading && !user) {
      navigate("/login");
    }
  }, [user, loading, navigate]);

  return (
    <div className="app-container">
      <header>
        <div>
          <p className="eyebrow">AI-as-a-Service</p>
          <h1>Pick a workspace</h1>
          <p className="subtitle">Choose the data type; we tailor validation, training, and reports.</p>
        </div>
      </header>

      <section className="panel highlight">
        <div className="panel-header">
          <div>
            <h2>Choose a mode to continue</h2>
            <p className="muted">Clear, single action. You can always go back.</p>
          </div>
          <div className="badge">Start</div>
        </div>
        <div className="mode-grid wide">
          {cards.map((card) => (
            <button
              key={card.key}
              className="mode-card active tight"
              type="button"
              onClick={() => navigate(`/workspace/${card.key}`)}
            >
              <div className="mode-card-content">
                <div className="mode-card-icon">
                  {card.key === "csv" ? (
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="1.5"/>
                      <line x1="9" y1="3" x2="9" y2="21" stroke="currentColor" strokeWidth="1.5"/>
                      <line x1="15" y1="3" x2="15" y2="21" stroke="currentColor" strokeWidth="1.5"/>
                      <line x1="3" y1="9" x2="21" y2="9" stroke="currentColor" strokeWidth="1.5"/>
                      <line x1="3" y1="15" x2="21" y2="15" stroke="currentColor" strokeWidth="1.5"/>
                    </svg>
                  ) : (
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <rect x="4" y="4" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.5"/>
                      <circle cx="9" cy="9" r="1.5" fill="currentColor"/>
                      <path d="M4 16L8 12L12 16L20 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  )}
                </div>
                <h3 className="mode-card-title">{card.title}</h3>
              </div>
              <div className="cta-inline">Use this mode →</div>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}

