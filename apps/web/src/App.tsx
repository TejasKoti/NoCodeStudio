import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import type { ReactNode } from "react";
import { useAuth } from "./store/auth";
import Workspace from "./pages/Workspace";
import Builder from "./pages/Builder";
import Login from "./pages/Login";
import LandingPage from "./pages/Landing";

function RequireAuth({ children }: { children: ReactNode }) {
  const { token } = useAuth();
  if (!token) return <Navigate to="/" replace />;
  return children;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<Login />} />

        {/* Protected routes */}
        <Route
          path="/projects"
          element={
            <RequireAuth>
              <Workspace />
            </RequireAuth>
          }
        />
        <Route
          path="/app/:id"
          element={
            <RequireAuth>
              <Builder />
            </RequireAuth>
          }
        />

        {/* Redirect unknown routes to landing */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}