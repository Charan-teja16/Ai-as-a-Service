import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import NavBar from "./components/NavBar";
import { AuthProvider } from "./contexts/AuthContext";
import App from "./App";
import ModeSelect from "./pages/ModeSelect";
import Login from "./pages/Login";
import Register from "./pages/Register";
import ForgotPassword from "./pages/ForgotPassword";
import ReportPreview from "./pages/ReportPreview";
import Profile from "./pages/Profile";
import ModelInfo from "./pages/ModelInfo";
import ModelDetail from "./pages/ModelDetail";
import "./style.css";
import { Toaster } from "react-hot-toast";

function AppShell() {
  const location = useLocation();
  const hideNav =
    location.pathname.startsWith("/login") ||
    location.pathname.startsWith("/register") ||
    location.pathname.startsWith("/forgot-password");
  return (
    <>
      {!hideNav && <NavBar />}
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/report/:reportId" element={<ReportPreview />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/models" element={<ModelInfo />} />
        <Route path="/models/:modelKey" element={<ModelDetail />} />
        <Route path="/" element={<ModeSelect />} />
        <Route path="/workspace/:mode" element={<App />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <Toaster position="top-right" />
    </>
  );
}

ReactDOM.createRoot(document.getElementById("app") as HTMLElement).render(
  <React.StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <AppShell />
      </AuthProvider>
    </BrowserRouter>
  </React.StrictMode>
);

