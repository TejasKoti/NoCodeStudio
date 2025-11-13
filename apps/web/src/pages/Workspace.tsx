import { useEffect, useMemo, useState, useRef } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../store/auth";
import {
  MoreVertical,
  User as UserIcon,
  LogOut,
  Folder as FolderIcon,
  ChevronLeft,
  ChevronRight,
  Home,
  Trash2,
  LayoutTemplate,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import logo from "../assets/Logo.png";
import placeholderThumb from "../assets/Empty_Canvas.png";

type Project = {
  _id: string;
  title: string;
  thumbnail?: string | null;
  updatedAt?: string;
  createdAt?: string;
  folderId?: string | null;
};


type Folder = {
  _id: string;
  name: string;
};

type Template = {
  _id: string;
  name: string;
};

export default function Workspace() {
  const navigate = useNavigate();
  const { user, token, logout } = useAuth();

  const [projects, setProjects] = useState<Project[]>([]);
  const [folders, setFolders] = useState<Folder[]>([]);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);
  const [creatingProject, setCreatingProject] = useState(false);
  const [creatingFolder, setCreatingFolder] = useState(false);
  const [newName, setNewName] = useState("");
  const [userOpen, setUserOpen] = useState(false);
  const userBtnRef = useRef<HTMLButtonElement | null>(null);

  const [selectedFolder, setSelectedFolder] = useState<string>("all");
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [sortMode, setSortMode] = useState<"modified" | "created" | "name" | "reverse">("modified");
  const [sortOpen, setSortOpen] = useState(false);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");
  const sortRef = useRef<HTMLDivElement | null>(null);
  const [showAddToFolderModal, setShowAddToFolderModal] = useState(false);
  const [selectedProjectToAdd, setSelectedProjectToAdd] = useState<any>(null);
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);


  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        sortRef.current &&
        !sortRef.current.contains(e.target as Node)
      ) {
        setSortOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Guard: must be logged in
  useEffect(() => {
    if (!token) navigate("/login");
  }, [token, navigate]);

  // Title
  useEffect(() => {
    document.title = "Workspace";
    let link =
      (document.querySelector("link[rel='icon']") as HTMLLinkElement) ||
      document.createElement("link");
    link.rel = "icon";
    link.href = logo;
    document.head.appendChild(link);
  }, []);

  // Fetch folders, projects, templates
  useEffect(() => {
    if (!token) return;
    const fetchAll = async () => {
      try {
        setLoading(true);
        const [projRes, foldRes, tempRes] = await Promise.all([
          axios.get("/api/projects", { headers: { Authorization: `Bearer ${token}` } }),
          axios.get("/api/folders", { headers: { Authorization: `Bearer ${token}` } }),
          axios.get("/api/templates", { headers: { Authorization: `Bearer ${token}` } }),
        ]);
        setProjects(projRes.data.projects || []);
        setFolders([{ _id: "all", name: "All Projects" }, ...(foldRes.data.folders || [])]);
        setTemplates(tempRes.data.templates || []);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    fetchAll();
  }, [token]);

  const updateScrollButtons = () => {
    if (!scrollRef.current) return;
    const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current;
    setCanScrollLeft(scrollLeft > 5);
    setCanScrollRight(scrollLeft + clientWidth < scrollWidth - 5);
  };

  const scrollByAmount = (dir: "left" | "right") => {
    if (!scrollRef.current) return;
    const scrollAmount = scrollRef.current.clientWidth * 0.8;
    scrollRef.current.scrollBy({
      left: dir === "left" ? -scrollAmount : scrollAmount,
      behavior: "smooth",
    });
  };

  useEffect(() => {
    const ref = scrollRef.current;
    if (!ref) return;
    ref.addEventListener("scroll", updateScrollButtons);
    updateScrollButtons();
    return () => ref.removeEventListener("scroll", updateScrollButtons);
  }, []);

  // Time formatting
  const timeAgo = (iso?: string) => {
    if (!iso) return "";
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins} min${mins > 1 ? "s" : ""} ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours} hour${hours > 1 ? "s" : ""} ago`;
    const days = Math.floor(hours / 24);
    return `${days} day${days > 1 ? "s" : ""} ago`;
  };

  // Create folder
  const createFolder = async () => {
    if (!newName.trim()) return;
    try {
      const res = await axios.post(
        "/api/folders",
        { name: newName.trim() },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setFolders((prev) => [...prev, res.data.folder]);
      setCreatingFolder(false);
      setNewName("");
    } catch (e) {
      console.error("Create folder error:", e);
    }
  };

  // Create project
  const createProject = async () => {
    try {
      const finalName = newName.trim() || "Untitled Project";
      const res = await axios.post(
        "/api/projects",
        {
          title: finalName,
          folderId: selectedFolder !== "all" ? selectedFolder : null,
          graph: { nodes: [], edges: [] },
        },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setCreatingProject(false);
      setNewName("");
      navigate(`/app/${res.data.project._id}`);
    } catch (e: any) {
      console.error("Create project error:", e);
      alert("Failed to create project — see console for details.");
    }
  };

  // Delete handlers with modal
  const [deletingItem, setDeletingItem] = useState<{ id: string; type: "project" | "folder"; name: string } | null>(null);

  const confirmDelete = async () => {
    if (!deletingItem) return;
    const { id, type } = deletingItem;

    try {
      if (type === "project") {
        await axios.delete(`/api/projects/${id}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setProjects((prev) => prev.filter((p) => p._id !== id));
      } else {
        await axios.delete(`/api/folders/${id}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setFolders((prev) => prev.filter((f) => f._id !== id));
        if (selectedFolder === id) setSelectedFolder("all");
      }
    } catch (e) {
      console.error("Delete error:", e);
    } finally {
      setDeletingItem(null);
    }
  };

  const deleteProject = (id: string, name: string) => {
    setDeletingItem({ id, type: "project", name });
  };

  const deleteFolder = (id: string, name: string) => {
    setDeletingItem({ id, type: "folder", name });
  };

  // Click outside user menu
  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      const t = e.target as HTMLElement;
      if (
        userBtnRef.current &&
        !t.closest?.("[data-user-menu]") &&
        t !== userBtnRef.current
      )
        setUserOpen(false);
    };
    document.addEventListener("click", onDocClick);
    return () => document.removeEventListener("click", onDocClick);
  }, []);

  // Sorting
  const sortedProjects = useMemo(() => {
    let arr = [...projects];
    if (selectedFolder !== "all")
      arr = arr.filter((p) => p.folderId === selectedFolder);

    const direction = sortDirection === "asc" ? 1 : -1;

    switch (sortMode) {
      case "name":
        arr.sort((a, b) => a.title.localeCompare(b.title) * direction);
        break;
      case "created":
        arr.sort(
          (a, b) =>
            ((a.createdAt && b.createdAt
              ? new Date(a.createdAt).getTime() -
                new Date(b.createdAt).getTime()
              : 0) * direction)
        );
        break;
      case "modified":
      default:
        arr.sort(
          (a, b) =>
            ((a.updatedAt && b.updatedAt
              ? new Date(a.updatedAt).getTime() -
                new Date(b.updatedAt).getTime()
              : 0) * direction)
        );
        break;
    }

    return arr;
  }, [projects, selectedFolder, sortMode, sortDirection]);

  // Grid
  const gridStyle = useMemo<React.CSSProperties>(
    () => ({
      display: "grid",
      gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
      gap: 24,
    }),
    []
  );

  // JSX
  return (
    <div style={{ display: "flex", height: "100vh", background: "linear-gradient(180deg, #0f0f0f 0%, #151515 100%)" }}>
      {/* LEFT SIDEBAR */}
      <div
        style={{
          position: "fixed",
          left: 0,
          top: 0,
          bottom: 0,
          width: 72,
          borderRight: "1px solid rgba(255,255,255,0.1)",
          background: "rgba(20,20,20,0.95)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          paddingTop: 20,
          gap: 16,
          zIndex: 10,
        }}
      >
        {/* Home Button */}
        <button
          onClick={() => window.location.reload()}
          title="Home"
          style={{
            width: 56,
            height: 56,
            borderRadius: 12,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(255,255,255,0.1)",
            border: "1px solid rgba(255,255,255,0.15)",
            cursor: "pointer",
            transition: "all 0.2s ease",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.2)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.1)")}
        >
          <Home
            color="white"
            strokeWidth={2.4}
            style={{
              width: "25px",
              height: "25px",
              flexShrink: 0,
              minWidth: "25px",
              minHeight: "25px",
            }}
          />
        </button>

        {/* Templates Button */}
        <button
          onClick={() => console.log("Templates section clicked")}
          title="Templates"
          style={{
            width: 56,
            height: 56,
            borderRadius: 12,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(255,255,255,0.1)",
            border: "1px solid rgba(255,255,255,0.15)",
            cursor: "pointer",
            transition: "all 0.2s ease",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.2)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.1)")}
        >
          <LayoutTemplate
            color="white"
            strokeWidth={2.3}
            style={{
              width: "25px",
              height: "25px",
              flexShrink: 0,
              minWidth: "25px",
              minHeight: "25px",
            }}
          />
        </button>
      </div>

      {/* MAIN CONTENT */}
      <div
        style={{
          flex: 1,
          height: "100vh",
          overflowY: "auto",
          background: "linear-gradient(180deg, #0f0f0f 0%, #151515 100%)",
          padding: "32px 48px",
          paddingLeft: "120px",
          display: "flex",
          flexDirection: "column",
          boxSizing: "border-box",
        }}
      >
        {/* HEADER */}
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 32,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <img src={logo} alt="Logo" width={28} height={28} style={{ borderRadius: 6 }} />
            <div style={{ fontWeight: 700, fontSize: 18 }}>Workspace</div>
          </div>
          <div>
            <button ref={userBtnRef} onClick={() => setUserOpen((v) => !v)} data-user-menu
              style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "6px 10px",
                borderRadius: 10,
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.05)",
                cursor: "pointer",
              }}>
              <UserIcon size={20} />
              <MoreVertical size={18} />
            </button>
            {userOpen && (
              <div style={{
                position: "absolute",
                top: 60, right: 60,
                padding: 12,
                borderRadius: 10,
                background: "rgba(40,40,40,0.9)",
                border: "1px solid rgba(255,255,255,0.12)",
              }}>
                <div style={{ fontSize: 12, opacity: 0.8 }}>Signed in as</div>
                <div style={{ fontWeight: 700, marginBottom: 8 }}>{user?.email}</div>
                <button onClick={() => { logout(); navigate("/login"); }}
                  style={{
                    display: "flex", alignItems: "center", gap: 8,
                    border: "1px solid rgba(255,255,255,0.12)",
                    padding: "8px 10px",
                    borderRadius: 8,
                    cursor: "pointer",
                    background: "transparent",
                    color: "white",
                  }}>
                  <LogOut size={18} /> Logout
                </button>
              </div>
            )}
          </div>
        </div>

        {/* FOLDERS */}
        <div style={{ marginBottom: 32 }}>
          <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 12 }}>Folders</div>

          <div
            style={{
              position: "relative",
              padding: "12px 80px",
              borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.1)",
              overflow: "hidden",
            }}
          >
            {/* Left scroll button */}
            <div
              onClick={() => scrollByAmount("left")}
              style={{
                position: "absolute",
                left: 12,
                top: "50%",
                transform: "translateY(-50%)",
                width: 50,
                height: 50,
                borderRadius: "50%",
                background: canScrollLeft
                  ? "rgba(255,255,255,0.3)"
                  : "rgba(255,255,255,0.1)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: canScrollLeft ? "pointer" : "default",
                opacity: canScrollLeft ? 1 : 0.4,
                zIndex: 50,
                pointerEvents: "auto",
                transition: "all 0.25s ease",
                boxShadow: canScrollLeft
                  ? "0 0 12px rgba(255,255,255,0.5)"
                  : "none",
              }}
              onMouseEnter={(e) => {
                if (canScrollLeft)
                  (e.currentTarget.style.boxShadow =
                    "0 0 10px rgba(255,255,255,0.9)");
              }}
              onMouseLeave={(e) => {
                (e.currentTarget.style.boxShadow = canScrollLeft
                  ? "0 0 6px rgba(255,255,255,0.7)"
                  : "none");
              }}
            >
              <ChevronLeft size={32} color="#fff" strokeWidth={3} />
            </div>

            {/* Scrollable folders row */}
            <div
              ref={scrollRef}
              style={{
                display: "flex",
                overflowX: "scroll",
                scrollbarWidth: "none",
                gap: 16,
                scrollBehavior: "smooth",
                paddingBottom: 8,
                pointerEvents: "auto",
              }}
              onWheel={(e) => {
                e.preventDefault();
              }}
              onScroll={updateScrollButtons}
            >
              {/* Create Folder */}
              <div
                onClick={() => setCreatingFolder(true)}
                style={{
                  flex: "0 0 160px",
                  height: 140,
                  borderRadius: 12,
                  border: "1px dashed rgba(255,255,255,0.2)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "pointer",
                  fontSize: 32,
                }}
              >
                ＋
              </div>

              {folders.map((f) => (
                <div
                  key={f._id}
                  style={{
                    flex: "0 0 160px",
                    height: 140,
                    borderRadius: 12,
                    border:
                      selectedFolder === f._id
                        ? "2px solid rgba(255,255,255,0.7)"
                        : "1px solid rgba(255,255,255,0.1)",
                    background: "rgba(255,255,255,0.04)",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    position: "relative",
                    cursor: "pointer",
                  }}
                  onClick={() => setSelectedFolder(f._id)}
                >
                  <FolderIcon size={26} />
                  <div style={{ marginTop: 8, fontWeight: 600 }}>{f.name}</div>
                  {f._id !== "all" && (
                    <div style={{ position: "absolute", bottom: 6, right: 6 }}>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteFolder(f._id, f.name);
                        }}
                        style={{
                          background: "rgba(255,0,0,0.1)",
                          border: "1px solid rgba(255,0,0,0.3)",
                          borderRadius: 6,
                          cursor: "pointer",
                          padding: 2,
                        }}
                      >
                        <Trash2 size={14} color="red" />
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Right scroll button */}
            <div
              onClick={() => scrollByAmount("right")}
              style={{
                position: "absolute",
                right: 12,
                top: "50%",
                transform: "translateY(-50%)",
                width: 50,
                height: 50,
                borderRadius: "50%",
                background: canScrollRight
                  ? "rgba(255,255,255,0.3)"
                  : "rgba(255,255,255,0.1)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: canScrollRight ? "pointer" : "default",
                opacity: canScrollRight ? 1 : 0.4,
                zIndex: 50,
                pointerEvents: "auto",
                transition: "all 0.25s ease",
                boxShadow: canScrollRight
                  ? "0 0 12px rgba(255,255,255,0.5)"
                  : "none",
              }}
              onMouseEnter={(e) => {
                if (canScrollRight)
                  (e.currentTarget.style.boxShadow =
                    "0 0 10px rgba(255,255,255,0.9)");
              }}
              onMouseLeave={(e) => {
                (e.currentTarget.style.boxShadow = canScrollRight
                  ? "0 0 6px rgba(255,255,255,0.7)"
                  : "none");
              }}
            >
              <ChevronRight size={32} color="#fff" strokeWidth={3} />
            </div>
          </div>
        </div>

        {/* PROJECTS */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 12,
            position: "relative",
          }}
        >
          <div style={{ fontSize: 20, fontWeight: 700 }}>Projects</div>

          {/* Sort Dropdown */}
          <div style={{ position: "relative" }} ref={sortRef}>
            <button
              onClick={() => setSortOpen((v) => !v)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 10px",
                borderRadius: 10,
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.05)",
                cursor: "pointer",
                color: "white",
              }}
            >
              Sort <MoreVertical size={16} />
            </button>

            {sortOpen && (
              <div
                style={{
                  position: "absolute",
                  right: 0,
                  top: 44,
                  background: "rgba(40,40,40,0.95)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 10,
                  padding: 10,
                  display: "flex",
                  flexDirection: "row",
                  alignItems: "stretch",
                  gap: 6,
                  zIndex: 1000,
                }}
              >
                {/* Sort options column */}
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 6,
                  }}
                >
                  {[
                    { label: "Modified", key: "modified" },
                    { label: "Created", key: "created" },
                    { label: "Name", key: "name" },
                  ].map((opt) => (
                    <button
                      key={opt.key}
                      onClick={() => setSortMode(opt.key as any)}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        padding: "6px 10px",
                        borderRadius: 8,
                        border: "1px solid rgba(255,255,255,0.2)",
                        background:
                          sortMode === opt.key
                            ? "rgba(255,255,255,0.15)"
                            : "rgba(255,255,255,0.05)",
                        color: "white",
                        cursor: "pointer",
                        fontSize: 14,
                        width: 120,
                      }}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>

                <button
                  onClick={() =>
                    setSortDirection((d) => (d === "asc" ? "desc" : "asc"))
                  }
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: "6px 10px",
                    borderRadius: 8,
                    border: "1px solid rgba(255,255,255,0.2)",
                    background: "rgba(255,255,255,0.08)",
                    color: "white",
                    cursor: "pointer",
                    fontSize: 16,
                    width: 36,
                  }}
                  title={`Sort ${sortDirection === "asc" ? "ascending" : "descending"}`}
                >
                  {sortDirection === "asc" ? "▲" : "▼"}
                </button>
              </div>
            )}
          </div>
        </div>

          <div style={gridStyle}>
            {/* + Create Project (only first row) */}
            <button onClick={() => setCreatingProject(true)}
              style={{
                height: 220,
                borderRadius: 16,
                border: "1px dashed rgba(255,255,255,0.18)",
                background: "rgba(255,255,255,0.05)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                fontSize: 40,
              }}>
              ＋
            </button>

            {loading ? (
              <div>Loading...</div>
            ) : sortedProjects.length === 0 ? (
              <div>No projects yet.</div>
            ) : (
              sortedProjects.map((p) => (
                <div
                  key={p._id}
                  style={{
                    borderRadius: 16,
                    overflow: "hidden",
                    border: "1px solid rgba(255,255,255,0.12)",
                    background: "rgba(255,255,255,0.05)",
                    position: "relative",
                    cursor: "pointer",
                  }}
                  onClick={() => navigate(`/app/${p._id}`)}
                >
                  {p.thumbnail ? (
                    <img
                      src={p.thumbnail}
                      alt={p.title}
                      style={{
                        width: "100%",
                        height: 160,
                        objectFit: "cover",
                        display: "block",
                      }}
                    />
                  ) : (
                    <div
                      style={{
                        height: 160,
                        backgroundImage: `url("${placeholderThumb}")`,
                        backgroundSize: "cover",
                        backgroundPosition: "center",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "rgba(255,255,255,0.5)",
                        fontSize: 14,
                        fontWeight: 500,
                      }}
                    >
                      No Thumbnail
                    </div>
                  )}
                  <div style={{ padding: 12 }}>
                    <div style={{ fontWeight: 600 }}>{p.title}</div>
                    <div style={{ fontSize: 12, opacity: 0.7 }}>
                      Updated {timeAgo(p.updatedAt)}
                    </div>
                  </div>

                  {/* Add + Delete Buttons */}
                  <div
                    style={{
                      position: "absolute",
                      bottom: 8,
                      right: 8,
                      display: "flex",
                      gap: 6,
                    }}
                  >
                    {/* Add to Folder */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedProjectToAdd(p);
                        setShowAddToFolderModal(true);
                      }}
                      style={{
                        background: "rgba(80,150,255,0.15)",
                        border: "1px solid rgba(80,150,255,0.35)",
                        borderRadius: 6,
                        cursor: "pointer",
                        padding: 2,
                        backdropFilter: "blur(4px)",
                      }}
                      title="Add to Folder"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="#5caeff"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <line x1="12" y1="5" x2="12" y2="19" />
                        <line x1="5" y1="12" x2="19" y2="12" />
                      </svg>
                    </button>

                    {/* Delete Project */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteProject(p._id, p.title);
                      }}
                      style={{
                        background: "rgba(255,0,0,0.1)",
                        border: "1px solid rgba(255,0,0,0.3)",
                        borderRadius: 6,
                        cursor: "pointer",
                        padding: 2,
                      }}
                      title="Delete Project"
                    >
                      <Trash2 size={14} color="red" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

      {/* MODALS */}
      <AnimatePresence>
        {showAddToFolderModal && selectedProjectToAdd && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(0,0,0,0.55)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 2200,
            }}
            onClick={() => setShowAddToFolderModal(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              style={{
                width: 440,
                maxHeight: "70vh",
                overflowY: "auto",
                borderRadius: 18,
                padding: "28px 24px",
                background: "rgba(40,40,40,0.9)",
                border: "1px solid rgba(255,255,255,0.15)",
                color: "white",
              }}
            >
              <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>
                Add “{selectedProjectToAdd.title}” to Folder
              </div>

              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                  maxHeight: 300,
                  overflowY: "auto",
                }}
              >
                {folders
                  .filter((f) => f._id !== "all")
                  .map((f) => (
                    <label
                      key={f._id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                        background: "rgba(255,255,255,0.05)",
                        padding: "10px 12px",
                        borderRadius: 10,
                        border: "1px solid rgba(255,255,255,0.1)",
                        cursor: "pointer",
                      }}
                    >
                      <input
                        type="radio"
                        checked={selectedFolderId === f._id}
                        onChange={() =>
                          setSelectedFolderId(
                            selectedFolderId === f._id ? null : f._id
                          )
                        }
                        style={{ width: 18, height: 18 }}
                      />
                      <FolderIcon size={18} />
                      <span>{f.name}</span>
                    </label>
                  ))}
              </div>

              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 10,
                  marginTop: 24,
                }}
              >
                <button
                  disabled={!selectedFolderId}
                  onClick={async () => {
                    if (!selectedFolderId) return;
                    try {
                      await axios.put(
                        `/api/folders/${selectedFolderId}/projects`,
                        { projectId: selectedProjectToAdd._id, action: "add" },
                        { headers: { Authorization: `Bearer ${token}` } }
                      );
                      // Update UI
                      setProjects((prev) =>
                        prev.map((proj) =>
                          proj._id === selectedProjectToAdd._id
                            ? { ...proj, folderId: selectedFolderId }
                            : proj
                        )
                      );
                      setShowAddToFolderModal(false);
                      setSelectedFolderId(null);
                    } catch (err) {
                      console.error("Add to folder error:", err);
                      alert("Failed to add project to folder.");
                    }
                  }}
                  style={{
                    flex: 1,
                    padding: "10px 12px",
                    borderRadius: 10,
                    border: "1px solid rgba(80,150,255,0.4)",
                    background: "rgba(50,120,255,0.25)",
                    fontWeight: 600,
                    color: "white",
                    cursor: selectedFolderId ? "pointer" : "not-allowed",
                    opacity: selectedFolderId ? 1 : 0.5,
                  }}
                >
                  Confirm
                </button>
                <button
                  onClick={() => setShowAddToFolderModal(false)}
                  style={{
                    flex: 1,
                    padding: "10px 12px",
                    borderRadius: 10,
                    border: "1px solid rgba(255,255,255,0.25)",
                    background: "rgba(255,255,255,0.1)",
                    fontWeight: 600,
                    color: "white",
                    cursor: "pointer",
                  }}
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {(creatingFolder || creatingProject) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(0,0,0,0.55)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 2000,
            }}
            onClick={() => {
              setCreatingFolder(false);
              setCreatingProject(false);
            }}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              style={{
                width: 420,
                borderRadius: 18,
                padding: "28px 24px",
                background: "rgba(40,40,40,0.9)",
                border: "1px solid rgba(255,255,255,0.15)",
                boxSizing: "border-box",
              }}
            >
              {/* Modal Title */}
              <div
                style={{
                  fontSize: 18,
                  fontWeight: 700,
                  marginBottom: 16,
                }}
              >
                {creatingFolder ? "Name your folder" : "Name your project"}
              </div>

              {/* Modal Body */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <input
                  autoFocus
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder={
                    creatingFolder ? "e.g. Vision Models" : "e.g. ConvNet v1"
                  }
                  onKeyDown={(e) =>
                    e.key === "Enter" &&
                    (creatingFolder ? createFolder() : createProject())
                  }
                  style={{
                    width: "100%",
                    padding: "12px 14px",
                    borderRadius: 10,
                    border: "1px solid rgba(255,255,255,0.12)",
                    background: "rgba(30,30,30,0.6)",
                    color: "white",
                    outline: "none",
                    fontSize: 14,
                    boxSizing: "border-box",
                  }}
                />

                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 10,
                  }}
                >
                  <button
                    onClick={creatingFolder ? createFolder : createProject}
                    style={{
                      flex: 1,
                      padding: "10px 12px",
                      borderRadius: 10,
                      border: "1px solid rgba(255,255,255,0.25)",
                      background: "rgba(255,255,255,0.25)",
                      fontWeight: 600,
                      color: "white",
                      cursor: "pointer",
                    }}
                  >
                    Create
                  </button>
                  <button
                    onClick={() => {
                      setCreatingFolder(false);
                      setCreatingProject(false);
                    }}
                    style={{
                      flex: 1,
                      padding: "10px 12px",
                      borderRadius: 10,
                      border: "1px solid rgba(255,255,255,0.25)",
                      background: "rgba(255,255,255,0.1)",
                      fontWeight: 600,
                      color: "white",
                      cursor: "pointer",
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      <AnimatePresence>
  {deletingItem && (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.55)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 2100,
      }}
      onClick={() => setDeletingItem(null)}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 420,
          borderRadius: 18,
          padding: "28px 24px",
          background: "rgba(40,40,40,0.9)",
          border: "1px solid rgba(255,255,255,0.15)",
          boxSizing: "border-box",
          color: "white",
        }}
      >
        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>
          Are you sure you want to delete this {deletingItem.type}?
        </div>

        <div
          style={{
            fontSize: 16,
            fontWeight: 600,
            marginBottom: 20,
            textAlign: "center",
            color: "rgba(255,255,255,0.9)",
          }}
        >
          {deletingItem.name}
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
          <button
            onClick={confirmDelete}
            style={{
              flex: 1,
              padding: "10px 12px",
              borderRadius: 10,
              border: "1px solid rgba(255,80,80,0.4)",
              background: "rgba(255,50,50,0.25)",
              fontWeight: 600,
              color: "white",
              cursor: "pointer",
            }}
          >
            Delete
          </button>
          <button
            onClick={() => setDeletingItem(null)}
            style={{
              flex: 1,
              padding: "10px 12px",
              borderRadius: 10,
              border: "1px solid rgba(80,150,255,0.4)",
              background: "rgba(50,120,255,0.25)",
              fontWeight: 600,
              color: "white",
              cursor: "pointer",
            }}
          >
            Cancel
          </button>
        </div>
      </motion.div>
    </motion.div>
  )}
</AnimatePresence>
    </div>
  );
}