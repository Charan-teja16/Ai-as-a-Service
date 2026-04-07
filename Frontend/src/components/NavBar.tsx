import { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import axios from "axios";
import toast from "react-hot-toast";
import { useAuth } from "../contexts/AuthContext";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export default function NavBar() {
  const { user, logout, refreshUser } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [subscribing, setSubscribing] = useState(false);

  const navItems = [
    { label: "Select Mode", to: "/" },
    { label: "Model Info", to: "/models" },
    { label: "Profile", to: "/profile" },
  ];

  const handleSubscribe = async () => {
    const confirmed = window.confirm(
      "Subscribe for unlimited training?\n\n" +
      "You will get:\n" +
      "• Unlimited model training\n" +
      "• No usage limits\n" +
      "• Full access to all features\n\n" +
      "This action cannot be undone. Continue?"
    );
    if (!confirmed) {
      return;
    }
    setSubscribing(true);
    try {
      const token = localStorage.getItem("token");
      await axios.post(
        `${API_BASE}/auth/subscribe`,
        { confirm: true },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      await refreshUser();
      toast.success("Successfully subscribed! You now have unlimited training.");
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || "Subscription failed");
    } finally {
      setSubscribing(false);
    }
  };

  return (
    <div className="nav-shell">
      <div className="nav-brand" onClick={() => navigate("/")}>
        <span className="nav-logo">AIaaS</span>
        <span className="nav-name">Studio</span>
      </div>
      <div className="nav-links">
        {navItems.map((item) => (
          <Link key={item.to} to={item.to} className={location.pathname.startsWith(item.to) ? "active" : ""}>
            {item.label}
          </Link>
        ))}
      </div>
      <div className="nav-user">
        <div className="nav-user-meta">
          <span className="nav-user-name">{user?.username ?? "Guest"}</span>
          <span className="nav-user-email">{user?.email ?? ""}</span>
          <span className="nav-user-sub">
            {user?.is_subscribed ? (
              <span className="badge success">Subscribed - Unlimited</span>
            ) : (
              <span className="badge">Free Runs: {user?.free_runs_remaining ?? 0} / 5</span>
            )}
          </span>
        </div>
        {user && !user.is_subscribed && (
          <button 
            className="nav-subscribe-btn" 
            onClick={handleSubscribe} 
            disabled={subscribing}
          >
            {subscribing ? "Subscribing..." : "Subscribe"}
          </button>
        )}
        {user ? (
          <button className="pill subtle" onClick={logout}>
            Logout
          </button>
        ) : (
          <button className="pill subtle" onClick={() => navigate("/login")}>
            Login
          </button>
        )}
      </div>
    </div>
  );
}

