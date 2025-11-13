import React, { useEffect } from "react";
import "remixicon/fonts/remixicon.css";
import logo from "../assets/Logo.png";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

export default function Landing() {
  const navigate = useNavigate();

  useEffect(() => {
    document.title = "NoCode Studio";
    const favicon =
      document.querySelector("link[rel='icon']") || document.createElement("link");
    favicon.setAttribute("rel", "icon");
    favicon.setAttribute("type", "image/png");
    favicon.setAttribute("href", logo);
    document.head.appendChild(favicon);
  }, []);

  return (
    <div
      style={{
        height: "100vh",
        width: "100vw",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        color: "white",
        position: "relative",
        textAlign: "center",
      }}
    >
      {/* Infinite dotted grid background */}
      <motion.div
        animate={{ backgroundPositionX: ["0%", "-200%"] }}
        transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
        style={{
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(rgba(255,255,255,0.07) 1px, transparent 1px)",
          backgroundSize: "24px 24px",
          backgroundColor: "#0a0a0a",
          zIndex: 0,
        }}
      />

      {/* Sun glow for eyecatching flashbang */}
      <motion.div
        animate={{
          opacity: [0.45, 0.9, 0.45],
          scale: [1, 1.08, 1],
          x: [0, -10, 0],
          y: [0, -10, 0],
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        style={{
          position: "absolute",
          right: "-25%",
          bottom: "-30%",
          width: "150%",
          height: "150%",
          background:
            "radial-gradient(circle at bottom right, rgba(255, 149, 0, 0.89) 0%, rgba(255, 119, 0, 0.72) 25%, rgba(0,0,0,0) 65%)",
          zIndex: 1,
          pointerEvents: "none",
        }}
      />

      {/* glowing halo behind logo */}
      <div style={{ position: "relative", zIndex: 3 }}>
        <motion.div
          animate={{ scale: [1, 1.15, 1], opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          style={{
            position: "absolute",
            inset: 0,
            margin: "auto",
            width: 260,
            height: 260,
            borderRadius: "50%",
            background:
              "radial-gradient(circle, rgba(255,150,0,0.55), transparent 60%)",
            filter: "blur(40px)",
            zIndex: 2,
          }}
        />

        <motion.img
          src={logo}
          alt="NoCode Logo"
          initial={{ opacity: 1, scale: 1 }}
          animate={{ scale: [1, 1.08, 1] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          style={{
            width: 200,
            height: 200,
            objectFit: "contain",
            position: "relative",
            zIndex: 3,
          }}
        />
      </div>

      {/* title */}
      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.8 }}
        style={{
          fontSize: "4.5rem",
          fontWeight: 900,
          marginTop: 36,
          zIndex: 3,
          background: "linear-gradient(90deg, #ff8c00, #ffb347, #fff)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}
      >
        NoCode Studio
      </motion.h1>

      {/* subtitle */}
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 0.8 }}
        style={{
          fontSize: "1.25rem",
          maxWidth: 520,
          lineHeight: 1.6,
          opacity: 0.9,
          marginTop: 14,
          zIndex: 3,
        }}
      >
        Create, train, and deploy AI visually — no code required.
      </motion.p>

      {/* buttons */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1, duration: 0.8 }}
        style={{
          display: "flex",
          gap: "20px",
          marginTop: 50,
          zIndex: 3,
        }}
      >
        <button
          onClick={() => navigate("/login")}
          style={{
            padding: "16px 36px",
            borderRadius: 14,
            border: "1px solid rgba(255,255,255,0.25)",
            background:
              "linear-gradient(135deg, rgba(255,140,0,0.9), rgba(255,180,80,0.9))",
            color: "#000",
            fontWeight: 700,
            fontSize: "18px",
            cursor: "pointer",
            transition: "all 0.25s ease",
            boxShadow: "0 10px 30px rgba(255,150,0,0.35)",
          }}
          onMouseEnter={(e) =>
            (e.currentTarget.style.boxShadow =
              "0 12px 36px rgba(255,150,0,0.6)")
          }
          onMouseLeave={(e) =>
            (e.currentTarget.style.boxShadow =
              "0 10px 30px rgba(255,150,0,0.35)")
          }
        >
          Get Started
        </button>

        <button
          onClick={() =>
            window.open("https://github.com/TejasKoti/NoCodeStudio", "_blank")
          }
          style={{
            padding: "16px 36px",
            borderRadius: 14,
            border: "1px solid rgba(255,255,255,0.25)",
            background: "rgba(255,255,255,0.08)",
            color: "white",
            fontWeight: 600,
            fontSize: "18px",
            cursor: "pointer",
            backdropFilter: "blur(8px)",
            transition: "all 0.25s ease",
          }}
          onMouseEnter={(e) =>
            (e.currentTarget.style.background = "rgba(255,255,255,0.15)")
          }
          onMouseLeave={(e) =>
            (e.currentTarget.style.background = "rgba(255,255,255,0.08)")
          }
        >
          Learn More
        </button>
      </motion.div>

      {/* footer */}
      <motion.footer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.4 }}
        style={{
          position: "absolute",
          bottom: 20,
          fontSize: 13,
          opacity: 0.7,
          zIndex: 3,
        }}
      >
        ©{new Date().getFullYear()} Tejas Koti
      </motion.footer>
    </div>
  );
}