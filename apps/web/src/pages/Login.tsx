import { useEffect, useState, useLayoutEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "../lib/axios";
import { useAuth } from "../store/auth";
import logo from "../assets/Logo.png";
import { Eye, EyeOff } from "lucide-react";

export default function Login() {
  const navigate = useNavigate();
  const { token, login } = useAuth();

  const [mode, setMode] = useState<"login" | "signup">("login");
  const [animating, setAnimating] = useState(false);
  const [visibleMode, setVisibleMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // container measurements
  const shellRef = useRef<HTMLDivElement>(null);
  const [halfWidth, setHalfWidth] = useState(600);

  useLayoutEffect(() => {
    const calc = () => {
      if (shellRef.current) {
        setHalfWidth(Math.round(shellRef.current.offsetWidth / 2));
      }
    };
    calc();
    window.addEventListener("resize", calc);
    return () => window.removeEventListener("resize", calc);
  }, []);

  const [toastPos, setToastPos] = useState<{ left: number; top: number } | null>(null);
  const updateToastPos = () => {
    if (!shellRef.current) return;
    const r = shellRef.current.getBoundingClientRect();
    setToastPos({ left: r.left + r.width / 2, top: r.top - 16 });
  };

  useLayoutEffect(() => {
    updateToastPos();
    const onResize = () => updateToastPos();
    const onScroll = () => updateToastPos();
    window.addEventListener("resize", onResize);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("resize", onResize);
      window.removeEventListener("scroll", onScroll);
    };
  }, [shellRef.current, mode, visibleMode, error]);

  useEffect(() => {
    if (token) navigate("/projects", { replace: true });
  }, [token, navigate]);

  const doSubmit = async () => {
    try {
      const url = visibleMode === "login" ? "/api/auth/login" : "/api/auth/register";
      const res = await api.post(url, { email, password });
      login(res.data.token, res.data.user);
      navigate("/projects");
    } catch (err: any) {
      setError(err?.response?.data?.message || "Invalid credentials. Please check again.");
    }
  };

  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 5000);
    return () => clearTimeout(t);
  }, [error]);

  const EyeButton = ({
    showPassword,
    setShowPassword,
  }: {
    showPassword: boolean;
    setShowPassword: React.Dispatch<React.SetStateAction<boolean>>;
  }) => {
    return (
      <button
        type="button"
        onClick={() => setShowPassword((v) => !v)}
        onMouseDown={(e) => e.preventDefault()}
        style={{
          position: "absolute",
          right: 6,
          top: "50%",
          transform: "translateY(-50%)",
          borderRadius: 8,
          border: "1px solid rgba(255, 255, 255, 0.25)",
          background: showPassword ? "rgba(255, 145, 0, 0.9)" : "rgba(255,255,255,0.08)",
          padding: 6,
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transition: "box-shadow .15s ease, background .15s ease",
          boxShadow: showPassword ? "0 0 10px rgba(255, 123, 0, 0.8)" : "none",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.boxShadow = showPassword
            ? "0 0 12px rgba(255, 141, 65, 0.9)"
            : "0 0 8px rgba(255, 145, 0, 0.4)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.boxShadow = showPassword
            ? "0 0 10px rgba(255, 174, 0, 0.8)"
            : "none";
        }}
      >
        {showPassword ? (
          <Eye size={18} color="black" strokeWidth={2} />
        ) : (
          <EyeOff size={18} color="white" strokeWidth={2} />
        )}
      </button>
    );
  };

  const inputBase = {
    width: "100%",
    padding: "12px 42px 12px 12px",
    borderRadius: 10,
    border: "1px solid rgba(255,255,255,0.25)",
    background: "rgba(255,255,255,0.08)",
    color: "white",
    outline: "none",
    transition: "all 0.25s ease",
  } as const;

  const glow = {
    boxShadow: "0 0 10px rgba(255, 150, 52, 0.6)",
    borderColor: "rgba(255, 145, 0, 0.8)",
  };

  const triggerSwitch = (newMode: "login" | "signup") => {
    if (animating) return;
    setAnimating(true);
    setMode(newMode);
    setTimeout(() => setVisibleMode(newMode), 400);
    setTimeout(() => setAnimating(false), 800);
  };

  const panelStyle: React.CSSProperties = {
    position: "absolute",
    top: 0,
    left: 0,
    width: "50%",
    height: "100%",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    backdropFilter: "blur(24px)",
    background: "linear-gradient(180deg, rgba(255,255,255,0.2), rgba(255,255,255,0.05))",
    zIndex: 3,
  };

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        color: "white",
        position: "relative",
      }}
    >
      {/* moving infinite grid bg */}
      <motion.div
        animate={{ backgroundPositionX: ["0%", "-200%"] }}
        transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
        style={{
          position: "absolute",
          inset: 0,
          background: "radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px)",
          backgroundSize: "24px 24px",
          backgroundColor: "#090909",
          zIndex: 0,
        }}
      />

      {/* shell */}
      <motion.div
        ref={shellRef}
        layout
        transition={{ duration: 0.8, ease: "easeInOut" }}
        style={{
          position: "relative",
          width: "70rem",
          maxWidth: "95vw",
          height: "38rem",
          borderRadius: 20,
          overflow: "hidden",
          boxShadow: `
            inset 0 0 3px rgba(0,0,0,0.9),
            0 0 2px rgba(255,255,255,0.25),
            0 0 10px rgba(255,255,255,0.12),
            0 2px 6px rgba(0,0,0,0.8)
          `,
          zIndex: 2,
        }}
      >
        {/* left panel (form) */}
        <motion.div
          animate={{ x: mode === "login" ? 0 : halfWidth }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
          style={{ ...panelStyle, left: 0 }}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={visibleMode}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              style={{
                width: "80%",
                maxWidth: 400,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                textAlign: "center",
              }}
            >
              <h2
                style={{
                  fontSize: 28,
                  fontWeight: 900,
                  marginBottom: 30,
                  textAlign: "center",
                }}
              >
                {visibleMode === "login" ? "User Login" : "Create Account"}
              </h2>

              <div
                style={{
                  width: "100%",
                  maxWidth: 360,
                  display: "flex",
                  flexDirection: "column",
                  gap: 12,
                  textAlign: "left",
                }}
              >
                <label style={{ fontSize: 13, opacity: 0.85 }}>Email</label>
                <input
                  style={{
                    ...inputBase,
                    ...(focusedField === "email" ? glow : {}),
                    width: "100%",
                    boxSizing: "border-box",
                  }}
                  onFocus={() => setFocusedField("email")}
                  onBlur={() => setFocusedField(null)}
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />

                <label style={{ fontSize: 13, opacity: 0.85 }}>Password</label>
                <div style={{ position: "relative", width: "100%" }}>
                  <input
                    style={{
                      ...inputBase,
                      ...(focusedField === "password" ? glow : {}),
                      width: "100%",
                      boxSizing: "border-box",
                    }}
                    onFocus={() => setFocusedField("password")}
                    onBlur={() => setFocusedField(null)}
                    type={showPassword ? "text" : "password"}
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                  <EyeButton showPassword={showPassword} setShowPassword={setShowPassword} />
                </div>

                <button
                  onClick={doSubmit}
                  style={{
                    width: "100%",
                    padding: "12px 14px",
                    borderRadius: 10,
                    background:
                      "linear-gradient(180deg, rgba(255,255,255,0.25), rgba(255,255,255,0.15))",
                    border: "1px solid rgba(255,255,255,0.25)",
                    color: "white",
                    fontWeight: 700,
                    cursor: "pointer",
                    boxSizing: "border-box",
                    marginTop: 6,
                  }}
                >
                  {visibleMode === "login" ? "Login" : "Create Account"}
                </button>
              </div>

              <div
                style={{
                  marginTop: 20,
                  fontSize: 13,
                  opacity: 0.85,
                  alignSelf: "flex-start",
                  textAlign: "left",
                  width: "100%",
                  maxWidth: 360,
                }}
              >
                {visibleMode === "login" ? (
                  <>
                    New to this page?{" "}
                    <button
                      onClick={() => triggerSwitch("signup")}
                      style={{
                        background: "none",
                        color: "white",
                        border: "none",
                        cursor: "pointer",
                        textDecoration: "underline",
                      }}
                    >
                      Sign up
                    </button>
                  </>
                ) : (
                  <>
                    Already have an account?{" "}
                    <button
                      onClick={() => triggerSwitch("login")}
                      style={{
                        background: "none",
                        color: "white",
                        border: "none",
                        cursor: "pointer",
                        textDecoration: "underline",
                      }}
                    >
                      Log in
                    </button>
                  </>
                )}
              </div>
            </motion.div>
          </AnimatePresence>
        </motion.div>

        {/* right panel logo shine */}
        <motion.div
          animate={{ x: mode === "login" ? 0 : -halfWidth }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
          style={{
            position: "absolute",
            right: 0,
            top: 0,
            width: "50%",
            height: "100%",
            background: "linear-gradient(270deg, #ff8c00b0, #ffb247a9)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            borderLeft: "1px solid rgba(255,255,255,0.08)",
            zIndex: 2,
          }}
        >
          <motion.img
            src={logo}
            alt="NoCode Logo"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.8 }}
            style={{ width: "200px", height: "200px", objectFit: "contain" }}
          />
          <h1 style={{ fontSize: 40, fontWeight: 900, marginTop: 16 }}>NoCode</h1>
          <p style={{ opacity: 0.8, fontSize: 14 }}>Build deep learning visually</p>
        </motion.div>
      </motion.div>

<AnimatePresence>
  {error && shellRef.current && (
    <motion.div
      key="error-toast"
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: -50, opacity: 0 }}
      transition={{
        duration: 0.5,
        ease: [0.25, 0.8, 0.5, 1],
      }}
      style={{
        position: "fixed",
        top: shellRef.current.getBoundingClientRect().top - 60,
        left:
          shellRef.current.getBoundingClientRect().left +
          shellRef.current.getBoundingClientRect().width / 2 - 90,
        transform: "translateX(-50%)",
        background: "rgba(255, 80, 80, 0.2)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(255, 80, 80, 0.4)",
        borderRadius: 12,
        padding: "10px 28px",
        color: "#ffcccc",
        fontWeight: 500,
        zIndex: 9999,
        pointerEvents: "none",
        whiteSpace: "nowrap",
        boxShadow: "0 4px 15px rgba(255, 80, 80, 0.2)",
      }}
    >
      {error}
    </motion.div>
  )}
</AnimatePresence>
    </div>
  );
}