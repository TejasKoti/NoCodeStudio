import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "../store/auth";
import { api } from "../lib/axios";
import html2canvas from "html2canvas";
import { motion, AnimatePresence } from "framer-motion";
import { LineChart, Line, XAxis, YAxis, Tooltip as ReTooltip } from "recharts";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
} from "@xyflow/react";
import type { Node, Edge, Connection, XYPosition, OnSelectionChangeParams } from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import {
  Save,
  Play,
  Download,
  ChevronLeft,
  ChevronRight,
  MoreVertical,
  User as UserIcon,
  Code2,
  FileJson,
  LogOut,
  Layers as LayersIcon,
  Upload,
  Activity,
  Settings,
  Trash2,
  BrainCircuit,
} from "lucide-react";

// Types
type RFNode = Node<{ label: string; params?: Record<string, any> }>;
type RFEdge = Edge;

function Tooltip({
  label,
  children,
  align = "bottom",
}: {
  label: string;
  children: React.ReactNode;
  align?: "top" | "bottom" | "left" | "right";
}) {
  const [open, setOpen] = useState(false);
  return (
    <div
      style={{ position: "relative", display: "inline-flex" }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      {children}
      {open && (
        <div
          style={{
            position: "absolute",
            whiteSpace: "nowrap",
            padding: "6px 10px",
            borderRadius: 10,
            backdropFilter: "blur(10px)",
            background: "rgba(20,20,20,0.6)",
            border: "1px solid rgba(255,255,255,0.15)",
            color: "white",
            fontSize: 12,
            top: align === "top" ? -36 : undefined,
            bottom: align === "bottom" ? -36 : undefined,
            left: align === "left" ? -36 : "50%",
            transform:
              align === "left"
                ? "translateX(-100%)"
                : align === "right"
                ? "translateX(0)"
                : "translate(-50%, 0)",
            boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
            pointerEvents: "none",
            zIndex: 4000,
          }}
        >
          {label}
        </div>
      )}
    </div>
  );
}

function IconButton({
  onClick,
  disabled,
  children,
  active,
}: {
  onClick?: () => void;
  disabled?: boolean;
  children: React.ReactNode;
  active?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: 8,
        borderRadius: 10,
        border: "1px solid rgba(255,255,255,0.12)",
        background: active
          ? "linear-gradient(180deg, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.10) 100%)"
          : "rgba(255,255,255,0.05)",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.6 : 1,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        transition: "transform .12s ease, box-shadow .12s ease, background .12s ease",
        boxShadow: active ? "0 8px 18px rgba(0,0,0,0.25)" : "none",
      }}
      onMouseEnter={(e) => {
        if (!disabled) {
          e.currentTarget.style.boxShadow = "0 8px 18px rgba(0,0,0,0.25)";
          e.currentTarget.style.transform = "translateY(-1px)";
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = "none";
        e.currentTarget.style.transform = "translateY(0)";
      }}
    >
      {children}
    </button>
  );
}

function Divider() {
  return (
    <div
      style={{
        width: 1,
        height: 28,
        background: "linear-gradient(180deg, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.08) 100%)",
        margin: "0 8px",
      }}
    />
  );
}

function BuilderInner() {
  const [alertBox, setAlertBox] = useState<{ message: string; type: "red" | "blue" } | null>(null);
  const { id } = useParams();
  const navigate = useNavigate();
  const { token, user, logout } = useAuth();

  const [projectTitle, setProjectTitle] = useState<string>("");
  const [isTitleEditing, setIsTitleEditing] = useState(false);

  const [nodes, setNodes, onNodesChange] = useNodesState<RFNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<RFEdge>([]);

  type LayerItem = { name: string; class: string | null; function: string | null };
  const [catalog, setCatalog] = useState<LayerItem[] | null>(null);

  const [selectedNode, setSelectedNode] = useState<RFNode | null>(null);
  const [selectedParams, setSelectedParams] = useState<any[] | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);

  // UI state
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [userOpen, setUserOpen] = useState(false);
  const [locked, setLocked] = useState(false);
  const rf = useReactFlow();
  const userBtnRef = useRef<HTMLButtonElement | null>(null);
  const [datasetMenuOpen, setDatasetMenuOpen] = useState(false);
  
  type MetricTab = "Loss" | "Speed" | "Time/Batch" | "ETA";

  interface TrainMetrics {
    loss?: number[];
    speed?: number[];
    batch_times?: number[];
    eta?: number[];
  }

  const [trainMetrics, setTrainMetrics] = useState<TrainMetrics | null>(null);
  const [activeTab, setActiveTab] = useState<"Loss" | "Speed" | "Time/Batch" | "ETA">("Loss");

  // Search state
  const [search, setSearch] = useState("");

  // Helpers
  const authHeader = useMemo(
    () => ({ headers: { Authorization: `Bearer ${token}` } }),
    [token]
  );

  // Derived: filtered catalog
  const filteredCatalog = useMemo(
    () =>
      (catalog || []).filter((op) =>
        op.name.toLowerCase().includes(search.toLowerCase())
      ),
    [catalog, search]
  );
  // Trainer / Dataset / Config states
  const [trainerFile, setTrainerFile] = useState<File | null>(null);
  const [trainerCode, setTrainerCode] = useState<string | null>(null);
  const [useCustomTrainer, setUseCustomTrainer] = useState(false);

  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState<string | null>(null);
  const [trainerUploadedPopup, setTrainerUploadedPopup] = useState<null | { name: string }>(null);
  const [datasetUploadedPopup, setDatasetUploadedPopup] = useState<null | { name: string }>(null);

  const [configOpen, setConfigOpen] = useState(false);
  const [trainerConfig, setTrainerConfig] = useState({
    Epochs: 3,
    LearningRate: 0.001,
    BatchSize: 32,
  });

  const [trainOutput, setTrainOutput] = useState<string>("");
  const [trainOverlay, setTrainOverlay] = useState(false);

  const [trainedModels, setTrainedModels] = useState<
    { name: string; data: string }[]
  >(() => {
    try {
      return JSON.parse(localStorage.getItem("trainedModels") || "[]");
    } catch {
      return [];
    }
  });

  // Left panel width (fully hidden when collapsed)
  const leftPanelWidth = leftCollapsed ? 0 : 260;

  // Bubble positions
  const bubbleGap = 12;
  const headerHeight = 56;
  const bubbleTop = headerHeight + 28;
  const [modelMenuOpen, setModelMenuOpen] = useState(false);

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem("trainedModels") || "[]");
    if (saved.length > 20) {
      localStorage.setItem("trainedModels", JSON.stringify(saved.slice(-10)));
    }
  }, []);

  // Load project + graph
  useEffect(() => {
    if (!token || !id) return;
    (async () => {
      try {
        setLoading(true);
        const res = await api.get(`/api/projects/${id}`, authHeader);
        const p = res.data.project;
        setProjectTitle(p.title ?? "Untitled");
        setNodes((p.graph?.nodes as RFNode[]) || []);
        setEdges((p.graph?.edges as RFEdge[]) || []);
      } catch (err) {
        console.error("Load project error:", err);
      } finally {
        setLoading(false);
      }
    })();
  }, [token, id, setNodes, setEdges, authHeader]);

  // load layer catalog for sidebar
  useEffect(() => {
    (async () => {
      try {
        const res = await api.get("/api/catalog");
        setCatalog(Array.isArray(res.data?.layers) ? res.data.layers : []);
      } catch (err) {
        console.error("Catalog error:", err);
        setCatalog([]);
      }
    })();
  }, []);

  // Document chrome
  useEffect(() => {
    document.title = projectTitle ? `Builder Mode` : "Builder";
    let link =
      (document.querySelector("link[rel='icon']") as HTMLLinkElement) ||
      document.createElement("link");
    link.rel = "icon";
    link.href = "/assets/Logo.png";
    document.head.appendChild(link);
  }, [projectTitle]);

  // Canvas interactions
  const onConnect = useCallback(
    (conn: Connection) => {
      setEdges((eds) => addEdge(conn, eds));
      setDirty(true);
    },
    [setEdges]
  );

  const [trainerMenuOpen, setTrainerMenuOpen] = useState(false);
const [uploadedTrainers, setUploadedTrainers] = useState<
  { name: string; code: string }[]
>(() => {
  try {
    return JSON.parse(localStorage.getItem("uploadedTrainers") || "[]");
  } catch {
    return [];
  }
});
const [selectedTrainer, setSelectedTrainer] = useState<string | null>(null);

const handleTrainerSelect = (name: string) => {
  const trainer = uploadedTrainers.find((t) => t.name === name);
  if (trainer) {
    setTrainerCode(trainer.code);
    setSelectedTrainer(name);
    setUseCustomTrainer(true);
  }
};

const handleTrainerDelete = (name: string) => {
  const next = uploadedTrainers.filter((t) => t.name !== name);
  setUploadedTrainers(next);
  localStorage.setItem("uploadedTrainers", JSON.stringify(next));
  if (selectedTrainer === name) {
    setSelectedTrainer(null);
    setTrainerCode(null);
    setUseCustomTrainer(false);
  }
};


  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const layer = event.dataTransfer.getData("application/layer");
      if (!layer) return;

      const canvas = event.currentTarget as HTMLElement;
      const paneRect = canvas.getBoundingClientRect();
      const pos: XYPosition = {
        x: event.clientX - paneRect.left - 120,
        y: event.clientY - paneRect.top - 20,
      };

      const newNode: RFNode = {
        id: `${Date.now()}`,
        position: pos,
        data: { label: layer, params: {} },
        type: "default",
      };

      setNodes((nds) => [...nds, newNode]);
      setDirty(true);
    },
    [setNodes]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onNodeClick = useCallback(
    async (_: any, node: RFNode) => {
      setSelectedNode(node);
      setSelectedNodeId(node.id);
      try {
        const res = await api.get(`/api/layer/${node.data.label}`);
        const params = (res.data.params || []).map((p: any) => ({
          name: p.name,
          value: p.default ?? "",
        }));
        setSelectedParams(params);
      } catch (e) {
        console.error("Params error:", e);
        setSelectedParams([]);
      }
    },
    []
  );

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
    setSelectedNodeId(null);
    setSelectedParams(null);
  }, []);

  const onSelectionChange = useCallback((p: OnSelectionChangeParams) => {
    const first = p.nodes?.[0] as RFNode | undefined;
    if (!first) {
      setSelectedNode(null);
      setSelectedNodeId(null);
      setSelectedParams(null);
      return;
    }
    setSelectedNode(first);
    setSelectedNodeId(first.id);
  }, []);

  const onNodesChanged = useCallback(
    (changes: any) => {
      onNodesChange(changes);
      setDirty(true);
    },
    [onNodesChange]
  );
  const onEdgesChanged = useCallback(
    (changes: any) => {
      onEdgesChange(changes);
      setDirty(true);
    },
    [onEdgesChange]
  );

  // Delete selected node on Del key
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Delete" || e.key === "Backspace") {
        if (selectedNodeId) {
          setNodes((nds) => nds.filter((n) => n.id !== selectedNodeId));
          setEdges((eds) => eds.filter((e) => e.source !== selectedNodeId && e.target !== selectedNodeId));
          setSelectedNode(null);
          setSelectedNodeId(null);
          setSelectedParams(null);
          setDirty(true);
        }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [selectedNodeId, setNodes, setEdges]);

    const captureAndSaveThumbnail = async (id: string) => {
      try {
        const flow = document.querySelector(".react-flow") as HTMLElement | null;
        if (!flow) {
          console.warn("No .react-flow element found for thumbnail capture.");
          return;
        }

        // Capture full builder view first
        const fullCanvas = await html2canvas(flow, {
          backgroundColor: "#111",
          scale: 1,
        });

        // Resize captured image to 400x200 since im poor to store high quality
        const targetWidth = 400;
        const targetHeight = 200;
        const resizedCanvas = document.createElement("canvas");
        resizedCanvas.width = targetWidth;
        resizedCanvas.height = targetHeight;
        const ctx = resizedCanvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(fullCanvas, 0, 0, targetWidth, targetHeight);
        }

        // Convert resized image to Base64
        const dataUrl = resizedCanvas.toDataURL("image/png");

        const token = localStorage.getItem("token");
        if (!token) return;

        const res = await api.put(
          `/api/projects/${id}`,
          { thumbnail: dataUrl },
          { headers: { Authorization: `Bearer ${token}` } }
        );

        if (res.status === 200) {
          console.log("Thumbnail saved!");
        } else {
          console.warn("Thumbnail save failed:", res.status);
        }
      } catch (err) {
        console.error("Thumbnail capture error:", err);
      }
    };

    const handleSave = async () => {
      try {
        setSaving(true);

        const token = localStorage.getItem("token");
        if (!token) {
          setAlertBox({ message: "No Auth Token found!", type: "red" });
          return;
        }

        const res = await api.put(
          `/api/projects/${id}`,
          { graph: { nodes, edges } },
          { headers: { Authorization: `Bearer ${token}` } }
        );

        if (res.status === 200) {
          console.log("Project saved:", res.data.project);
          setDirty(false);

          if (!id) {
            console.warn("No project id found for thumbnail capture.");
            return;
          }

          await captureAndSaveThumbnail(id);
        } else {
          console.error("Save failed:", res.status, res.data);
          setAlertBox({ message: "Save failed : Check logs for details", type: "red" });
        }
      } catch (e) {
        console.error("Save error:", e);
        setAlertBox({ message: "Save failed : Check logs for details", type: "red" });
      } finally {
        setSaving(false);
      }
    };

  // Title rename
  const commitTitle = useCallback(async () => {
    if (!id) return;
    try {
      await api.put(`/api/projects/${id}`, { title: projectTitle }, authHeader);
    } catch (e) {
      console.error("Rename error:", e);
    } finally {
      setIsTitleEditing(false);
    }
  }, [id, projectTitle, authHeader]);

  // Export (graph JSON)
  const exportGraphJSON = useCallback(() => {
    const dataStr = JSON.stringify({ nodes, edges }, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.download = `${projectTitle || "project"}_graph.json`;
    a.href = url;
    a.click();
    URL.revokeObjectURL(url);
  }, [nodes, edges, projectTitle]);

  // Export (simple PyTorch code)
  const exportCodePY = useCallback(async () => {
    try {
      const res = await api.post(
        "/api/export",
        {
          title: projectTitle || "MyModel",
          graph: { nodes, edges },
        },
        authHeader
      );

      const code = res.data.code || "# No code returned";
      const blob = new Blob([code], { type: "text/x-python" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.download = res.data.filename || `${projectTitle || "project"}.py`;
      a.href = url;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export error:", err);
      setAlertBox({ message: "Export failed : Check logs for details", type: "red" });
    }
  }, [nodes, edges, projectTitle, authHeader]);


  // Run (placeholder: tries /api/run if exists or kaboom)
  const [toast, setToast] = useState<string | null>(null);

  const generatePythonCode = async () => {
    const res = await api.post(
      "/api/export",
      { title: projectTitle || "MyModel", graph: { nodes, edges } },
      authHeader
    );
    return res.data.code || "# No code returned";
  };

  const handleRun = useCallback(async () => {
    try {
      setToast("Running model…"), setTimeout(() => setToast(null), 2000);

      const code = await generatePythonCode();

      const hasConv = nodes.some((n) => n.data.label?.toLowerCase().includes("conv"));
      const hasLinear = nodes.some((n) => n.data.label?.toLowerCase().includes("linear"));

      let input: any = null;
      if (hasConv) input = Array(1).fill(Array(3).fill(Array(32).fill(Array(32).fill(0))));
      else if (hasLinear) input = [[1, 2, 3, 4, 5, 6, 7, 8]];
      else input = [[1, 2, 3]];

      const res = await api.post("/api/run", { code, input }, authHeader);

      const { output, error, stdout } = res.data;

      if (error) {
        console.error("Python error:", error);
        setToast(`Run failed: ${error}`), setTimeout(() => setToast(null), 2000);
      } else if (output) {
        console.log("Model output:", output);
        setToast(`Output: ${JSON.stringify(output).slice(0, 100)}...`), setTimeout(() => setToast(null), 2000);
      } else if (stdout) {
        console.log("Python stdout:", stdout);
        setToast(`${stdout.slice(0, 150)}...`), setTimeout(() => setToast(null), 2000);
      } else {
        setToast("No output received from backend."), setTimeout(() => setToast(null), 2000);
      }
    } catch (err: any) {
      console.error("Run request error:", err);
      setToast(`Network or backend error: ${err?.response?.data?.error || err.message}`), setTimeout(() => setToast(null), 2000);
    } finally {
      setTimeout(() => setToast(null), 6000);
    }
  }, [nodes, edges, projectTitle, authHeader]);


  const handleTrain = useCallback(async () => {
    try {
      setToast("🚀 Training model…");
      setTimeout(() => setToast(null), 2000);

      setTrainOutput("");
      setTrainOverlay(true);
      setTrainMetrics(null);

      const code = await generatePythonCode();

      const body: any = {
        code,
        config: trainerConfig,
        trainer: useCustomTrainer ? trainerCode : null,
        dataset: datasetName || null,
      };

      const res = await api.post("/api/train", body, authHeader);
      const { error, stdout, metrics } = res.data;

      if (error) {
        console.error("TRAIN ERROR:", error);
        setTrainOutput(`Training failed:\n${error}`);
      } else {
        console.log("TRAIN STDOUT:", stdout);
        setTrainOutput(stdout || "Training complete!");
      }

      if (metrics) {
        setTrainMetrics({
          loss: metrics.loss || [],
          batch_times: metrics.batch_times || [],
          speed: metrics.speed || [],
          eta: metrics.eta || [],
        });
      }

      if (res.data?.modelBase64) {
        const modelName = `${projectTitle || "model"}_${new Date()
          .toISOString()
          .slice(0, 19)
          .replace(/[:T]/g, "-")}.pt`;

        const updated = [...trainedModels, { name: modelName, data: "" }].slice(-10);
        setTrainedModels(updated);
        const savedModels = JSON.parse(localStorage.getItem("trainedModels") || "[]");
        savedModels.push({ name: modelName, savedAt: new Date().toISOString() });
        localStorage.setItem("trainedModels", JSON.stringify(savedModels.slice(-10)));
        setToast(`Trained model saved: ${modelName}`), setTimeout(() => setToast(null), 2000);
        try {
          const b64 = res.data.modelBase64;
          const blob = new Blob(
            [Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))],
            { type: "application/octet-stream" }
          );

          const link = document.createElement("a");
          link.href = URL.createObjectURL(blob);
          link.download = modelName;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(link.href);
        } catch (err) {
          console.error("Download error:", err);
        }
      }

    } catch (err: any) {
      console.error("Train request error:", err);
      setTrainOutput(`Network error: ${err?.message}`);
    }
  }, [
    nodes,
    edges,
    projectTitle,
    trainerCode,
    trainerConfig,
    datasetName,
    useCustomTrainer,
    authHeader,
    trainedModels,
  ]);

  const renderMetricChart = (tab: MetricTab) => {
    if (!trainMetrics) {
      return <div style={{ color: "#888", fontSize: 14 }}>No data yet</div>;
    }

    let arr: number[] = [];

    switch (tab) {
      case "Loss": arr = trainMetrics.loss ?? []; break;
      case "Speed": arr = trainMetrics.speed ?? []; break;
      case "Time/Batch": arr = trainMetrics.batch_times ?? []; break;
      case "ETA": arr = trainMetrics.eta ?? []; break;
    }

    if (!arr || arr.length === 0) {
      return <div style={{ color: "#888", fontSize: 14 }}>No data available</div>;
    }

    // Recharts expects [{ step: 0, value: arr[0] }, ...till infinity]
    const data = arr.map((v, i) => ({ step: i, value: v }));

    return (
      <LineChart width={500} height={250} data={data}>
        <XAxis dataKey="step" />
        <YAxis />
        <ReTooltip />
        <Line type="monotone" dataKey="value" dot={false} strokeWidth={2} />
      </LineChart>
    );
  };

  // Param editing (local state in selected node)
  const updateParam = (name: string, value: string) => {
    if (!selectedNode) return;
    setSelectedParams((prev) =>
      (prev || []).map((p) => (p.name === name ? { ...p, value } : p))
    );
    // Also mirror into node.data.params so it's persisted in graph
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== selectedNode.id) return n;
        const nextParams = { ...(n.data?.params || {}), [name]: value };
        return { ...n, data: { ...n.data, params: nextParams } as any };
      })
    );
    setDirty(true);
  };

  // Close export / user dropdown on outside click
  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      const t = e.target as HTMLElement;
      if (!t.closest?.("[data-export-menu]")) setExportOpen(false);
      if (userBtnRef.current && !t.closest?.("[data-user-menu]") && t !== userBtnRef.current)
        setUserOpen(false);
    };
    document.addEventListener("click", onDocClick);
    return () => document.removeEventListener("click", onDocClick);
  }, []);

  if (!token) {
    return <div style={{ padding: 40 }}>Not authorized…</div>;
  }

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <style>{`
        @keyframes pulseSave {
          0% { box-shadow: 0 0 0 0 rgba(120, 200, 255, 0.45); }
          70% { box-shadow: 0 0 0 10px rgba(120, 200, 255, 0); }
          100% { box-shadow: 0 0 0 0 rgba(120, 200, 255, 0); }
        }
        body { overflow: hidden; }
      `}</style>
      <style>
        {`
        .react-flow__controls {
          gap: 2px !important;
        }

        .react-flow__controls-button {
          color: black !important;
          fill: black !important;
          stroke: black !important;
          background: rgba(255,255,255,0.85) !important;
          border: 1px solid rgba(0,0,0,0.15) !important;
        }
        .react-flow__node {
          background: rgba(255,255,255,0.12) !important;
          backdrop-filter: blur(14px);
          color: white !important;
          border: 1px solid rgba(255,255,255,0.18) !important;
        }

        .left-pane-scroll::-webkit-scrollbar {
          width: 6px;
        }
        .left-pane-scroll::-webkit-scrollbar-track {
          background: transparent;
        }
        .left-pane-scroll::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,0.18);
          border-radius: 6px;
        }
        .left-pane-scroll:hover::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,0.28);
        }
        .left-pane-scroll {
          scrollbar-width: thin;
          scrollbar-color: rgba(255,255,255,0.24) transparent;
        }
        .left-pane-scroll,
        .left-sidebar {
          overflow-x: hidden !important;
        }
        .glass-input {
          width: 100%;
          padding: 10px 12px;
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.18);
          background: rgba(255,255,255,0.06);
          backdrop-filter: blur(10px);
          color: white;
          outline: none;
        }
        .glass-input::placeholder {
          color: rgba(255,255,255,0.65);
        }
        .bubble-toggle {
          position: fixed;
          isolation: isolate;
          top: ${bubbleTop}px;
          z-index: 1000000;
          width: 36px;
          height: 36px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.28);
          background: rgba(20,20,20,0.45);
          backdrop-filter: blur(10px);
          display: inline-flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          box-shadow: 0 8px 22px rgba(0,0,0,0.35);
          transition: transform .16s ease, box-shadow .16s ease, background .16s ease, left .18s ease, top .18s ease;
          user-select: none;
          overflow: visible;
        }
        .bubble-toggle::after {
          content: "";
          position: absolute;
          top: 50%;
          left: 50%;
          width: 10px;
          height: 10px;
          border-right: 2.25px solid #fff;
          border-bottom: 2.25px solid #fff;
          transform: translate(calc(-50% + 2px), calc(-50% - 1px)) rotate(135deg);
          pointer-events: none;
        }
        .bubble-toggle[data-collapsed="true"]::after {
          transform: translate(calc(-50% - 2px), calc(-50% - 1px)) rotate(-45deg);
        }
        .bubble-toggle:hover {
          transform: scale(1.05);
          box-shadow: 0 10px 26px rgba(0,0,0,0.4);
          background: rgba(255,255,255,0.20);
        }
      `}
      </style>

      {/* Top Glass Bar (FIXED OVER CANVAS) */}
      <header
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          height: 56,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 12px",
          backdropFilter: "blur(10px)",
          background:
            "linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%)",
          borderBottom: "1px solid rgba(255,255,255,0.08)",
          gap: 8,
          zIndex: 500,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button
            onClick={() => navigate("/projects")}
            style={{
              padding: "8px 10px",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(255,255,255,0.05)",
              cursor: "pointer",
            }}
            title="Back to Workspace"
          >
            ← Workspace
          </button>

          {isTitleEditing ? (
            <input
              autoFocus
              value={projectTitle}
              onChange={(e) => setProjectTitle(e.target.value)}
              onBlur={commitTitle}
              onKeyDown={(e) => e.key === "Enter" && commitTitle()}
              style={{
                marginLeft: 8,
                padding: "6px 10px",
                borderRadius: 8,
                border: "1px solid rgba(255,255,255,0.18)",
                background: "rgba(0,0,0,0.25)",
                color: "white",
                outline: "none",
              }}
            />
          ) : (
            <div
              onDoubleClick={() => setIsTitleEditing(true)}
              style={{ fontWeight: 700, marginLeft: 8, cursor: "text" }}
              title="Double-click to rename"
            >
              {projectTitle || "Untitled"}
            </div>
          )}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            style={{
              fontSize: 12,
              opacity: dirty ? 1 : 0.75,
              padding: "6px 10px",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.12)",
              background: dirty ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.04)",
            }}
          >
            {dirty ? "Unsaved changes" : "All changes saved"}
          </div>

          <Tooltip label="Save">
            <span>
              <IconButton onClick={handleSave} disabled={!dirty || saving} active={saving}>
                <Save size={24} />
              </IconButton>
            </span>
          </Tooltip>

          <Tooltip label="Clear Cache">
            <span>
              <IconButton onClick={() => localStorage.removeItem("trainedModels")}>
                <Trash2 size={24} />
              </IconButton>
            </span>
          </Tooltip>

          <Divider />        

          {/* Export + Import menu */}
          <div style={{ position: "relative" }} data-export-menu>
            <Tooltip label="Export / Import">
              <span>
                <IconButton onClick={() => setExportOpen((v) => !v)}>
                  <Download size={24} />
                </IconButton>
              </span>
            </Tooltip>

            {exportOpen && (
              <div
                style={{
                  position: "absolute",
                  right: 0,
                  marginTop: 8,
                  minWidth: 240,
                  borderRadius: 12,
                  padding: 6,
                  background:
                    "linear-gradient(180deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.06) 100%)",
                  border: "1px solid rgba(255,255,255,0.15)",
                  backdropFilter: "blur(12px)",
                  boxShadow: "0 12px 28px rgba(0,0,0,0.35)",
                  zIndex: 600,
                }}
              >
                {/* Export Code (.py) */}
                <button
                  onClick={exportCodePY}
                  style={{
                    width: "100%",
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "10px 12px",
                    borderRadius: 10,
                    background: "transparent",
                    border: "none",
                    cursor: "pointer",
                    color: "white",
                  }}
                >
                  <Code2 size={18} /> Export Code (.py)
                </button>

                {/* Export Graph (.json) */}
                <button
                  onClick={exportGraphJSON}
                  style={{
                    width: "100%",
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "10px 12px",
                    borderRadius: 10,
                    background: "transparent",
                    border: "none",
                    cursor: "pointer",
                    color: "white",
                  }}
                >
                  <FileJson size={18} /> Export Graph (.json)
                </button>

                {/* Divider between export/import */}
                <div
                  style={{
                    height: 1,
                    background: "rgba(255,255,255,0.1)",
                    margin: "6px 0",
                  }}
                />

                {/* Import Code (.py) */}
                <button
                  onClick={async () => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = ".py";
                    input.onchange = async (e) => {
                      const file = (e.target as HTMLInputElement)?.files?.[0];
                      if (!file) return;

                      const formData = new FormData();
                      formData.append("file", file);

                      try {
                        const res = await api.post("/api/import", formData, {
                          headers: {
                            Authorization: `Bearer ${token}`,
                            "Content-Type": "multipart/form-data",
                          },
                        });

                        const graph = res.data.graph;
                        if (graph?.nodes && graph?.edges) {
                          setNodes(graph.nodes);
                          setEdges(graph.edges);
                          setDirty(true);
                          alert("Model imported into workspace!");
                          setAlertBox({ message: "Model imported into workspace!", type: "blue" });
                        } else {
                          alert("No graph returned from backend.");
                        }
                      } catch (err) {
                        console.error("Import error:", err);
                        setAlertBox({ message: "Import failed : Check logs for details", type: "red" });
                      }
                    };
                    input.click();
                  }}
                  style={{
                    width: "100%",
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "10px 12px",
                    borderRadius: 10,
                    background: "transparent",
                    border: "none",
                    cursor: "pointer",
                    color: "white",
                  }}
                >
                  <Upload size={18} /> Import Code (.py)
                </button>
              </div>
            )}
          </div>

          {/* Run button */}
          <Tooltip label="Run">
            <span>
              <IconButton onClick={handleRun}>
                <Play size={24} />
              </IconButton>
            </span>
          </Tooltip>
          {/* Trainer Switch Dropdown */}
          <div style={{ position: "relative" }} data-trainer-menu>
            <Tooltip label="Trainer Switch">
              <span>
                <IconButton onClick={() => setTrainerMenuOpen((v) => !v)}>
                  <BrainCircuit size={22} />
                </IconButton>
              </span>
            </Tooltip>

            {trainerMenuOpen && (
              <div
                style={{
                  position: "absolute",
                  right: 0,
                  marginTop: 8,
                  minWidth: 260,
                  borderRadius: 12,
                  padding: 8,
                  background:
                    "linear-gradient(180deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.06) 100%)",
                  border: "1px solid rgba(255,255,255,0.15)",
                  backdropFilter: "blur(12px)",
                  boxShadow: "0 12px 28px rgba(0,0,0,0.35)",
                  zIndex: 600,
                }}
              >
                {/* Default Trainer Entry */}
                <div
                  onClick={() => {
                    setUseCustomTrainer(false);
                    setSelectedTrainer(null);
                  }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "8px 10px",
                    borderRadius: 8,
                    marginBottom: 4,
                    background: !useCustomTrainer
                      ? "rgba(255,255,255,0.1)"
                      : "transparent",
                    cursor: "pointer",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    {/* Radio indicator */}
                    <div
                      style={{
                        width: 14,
                        height: 14,
                        borderRadius: "50%",
                        border: "2px solid white",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "transparent",
                      }}
                    >
                      {!useCustomTrainer && (
                        <div
                          style={{
                            width: 6,
                            height: 6,
                            borderRadius: "50%",
                            background: "white",
                          }}
                        />
                      )}
                    </div>
                    <div>Default Trainer</div>
                  </div>
                </div>

                {/* Uploaded Trainers */}
                {uploadedTrainers.length === 0 ? (
                  <div
                    style={{
                      opacity: 0.6,
                      fontSize: 13,
                      margin: "6px 0 10px 4px",
                    }}
                  >
                    No custom trainers yet
                  </div>
                ) : (
                  uploadedTrainers.map((t, idx) => (
                    <div
                      key={t.name + idx}
                      onClick={() => handleTrainerSelect(t.name)}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "8px 10px",
                        borderRadius: 8,
                        background:
                          useCustomTrainer && selectedTrainer === t.name
                            ? "rgba(255,255,255,0.1)"
                            : "transparent",
                        marginBottom: 4,
                        cursor: "pointer",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div
                          style={{
                            width: 14,
                            height: 14,
                            borderRadius: "50%",
                            border: "2px solid white",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            background: "transparent",
                          }}
                        >
                          {useCustomTrainer && selectedTrainer === t.name && (
                            <div
                              style={{
                                width: 6,
                                height: 6,
                                borderRadius: "50%",
                                background: "white",
                              }}
                            />
                          )}
                        </div>
                        <div>{t.name}</div>
                      </div>

                      {/* Delete Button (trash icon) */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleTrainerDelete(t.name);
                        }}
                        style={{
                          background: "rgba(255,0,0,0.08)",
                          border: "1px solid rgba(255,0,0,0.3)",
                          borderRadius: 6,
                          cursor: "pointer",
                          padding: 3,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                        title="Delete trainer"
                      >
                        <Trash2 size={15} color="rgb(255,80,80)" />
                      </button>
                    </div>
                  ))
                )}

                {/* Divider */}
                <div
                  style={{
                    height: 1,
                    background: "rgba(255,255,255,0.12)",
                    margin: "6px 0",
                  }}
                />

                {/* Upload Trainer Button */}
                <button
                  onClick={async () => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = ".py";
                    input.onchange = async (e) => {
                      const file = (e.target as HTMLInputElement)?.files?.[0];
                      if (!file) return;
                      const text = await file.text();

                      const trainerObj = { name: file.name, code: text };
                      const updated = [...uploadedTrainers, trainerObj];
                      setUploadedTrainers(updated);
                      localStorage.setItem("uploadedTrainers", JSON.stringify(updated));

                      setTrainerUploadedPopup({ name: file.name });
                    };
                    input.click();
                  }}
                  style={{
                    width: "100%",
                    padding: "10px 12px",
                    borderRadius: 10,
                    background: "rgba(255,255,255,0.08)",
                    border: "1px solid rgba(255,255,255,0.18)",
                    color: "white",
                    cursor: "pointer",
                    fontWeight: 600,
                  }}
                >
                  + Upload Trainer Script
                </button>
                <AnimatePresence>
                  {trainerUploadedPopup && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      style={{
                        position: "fixed",
                        inset: 0,
                        background: "rgba(0,0,0,0.45)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        zIndex: 2500,
                      }}
                      onClick={() => setTrainerUploadedPopup(null)}
                    >
                      <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.9, opacity: 0 }}
                        onClick={(e) => e.stopPropagation()}
                        style={{
                          width: 400,
                          borderRadius: 18,
                          padding: "26px 22px",
                          background: "rgba(30,30,35,0.95)",
                          border: "1px solid rgba(255,255,255,0.12)",
                          boxSizing: "border-box",
                          color: "white",
                          textAlign: "center",
                        }}
                      >
                        <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 14, color: "#6bd6ff" }}>
                          Trainer Uploaded
                        </div>

                        <div
                          style={{
                            fontSize: 15,
                            opacity: 0.9,
                            marginBottom: 24,
                            wordWrap: "break-word",
                          }}
                        >
                          <span style={{ color: "#bde9ff" }}>{trainerUploadedPopup.name}</span> was added successfully!
                        </div>

                        <button
                          onClick={() => setTrainerUploadedPopup(null)}
                          style={{
                            width: "100%",
                            padding: "10px 12px",
                            borderRadius: 10,
                            border: "1px solid rgba(80,150,255,0.4)",
                            background: "rgba(50,120,255,0.25)",
                            fontWeight: 600,
                            color: "white",
                            cursor: "pointer",
                          }}
                        >
                          OK
                        </button>
                      </motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
          </div>

          {/* Dataset Switch Dropdown */}
          <div style={{ position: "relative" }} data-dataset-menu>
            <Tooltip label="Dataset Switch">
              <span>
                <IconButton onClick={() => setDatasetMenuOpen((v) => !v)}>
                  <LayersIcon size={22} />
                </IconButton>
              </span>
            </Tooltip>

            {datasetMenuOpen && (
              <div
                style={{
                  position: "absolute",
                  right: 0,
                  marginTop: 8,
                  minWidth: 260,
                  borderRadius: 12,
                  padding: 8,
                  background:
                    "linear-gradient(180deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.06) 100%)",
                  border: "1px solid rgba(255,255,255,0.15)",
                  backdropFilter: "blur(12px)",
                  boxShadow: "0 12px 28px rgba(0,0,0,0.35)",
                  zIndex: 600,
                }}
              >
                {/* Built-in Datasets */}
                {["CIFAR10", "MNIST"].map((name) => (
                  <div
                    key={name}
                    onClick={() => {
                      setDatasetName(name);
                      setDatasetFile(null);
                      localStorage.setItem("customDataset", "");
                    }}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "8px 10px",
                      borderRadius: 8,
                      marginBottom: 4,
                      background:
                        datasetName === name && !datasetFile
                          ? "rgba(255,255,255,0.1)"
                          : "transparent",
                      cursor: "pointer",
                      transition: "background 0.2s ease",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      {/* Radio indicator */}
                      <div
                        style={{
                          width: 16,
                          height: 16,
                          borderRadius: "50%",
                          border: "2px solid white",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          background: "transparent",
                        }}
                      >
                        {datasetName === name && !datasetFile && (
                          <div
                            style={{
                              width: 8,
                              height: 8,
                              borderRadius: "50%",
                              background: "white",
                            }}
                          />
                        )}
                      </div>
                      <div>{name}</div>
                    </div>
                  </div>
                ))}

                {/* Uploaded Custom Dataset */}
                {datasetFile ? (
                  <div
                    onClick={() => {
                      setDatasetName(datasetFile.name);
                    }}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "8px 10px",
                      borderRadius: 8,
                      marginBottom: 4,
                      background:
                        datasetFile && datasetName === datasetFile.name
                          ? "rgba(255,255,255,0.1)"
                          : "transparent",
                      cursor: "pointer",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div
                        style={{
                          width: 16,
                          height: 16,
                          borderRadius: "50%",
                          border: "2px solid white",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          background: "transparent",
                        }}
                      >
                        {datasetFile && datasetName === datasetFile.name && (
                          <div
                            style={{
                              width: 8,
                              height: 8,
                              borderRadius: "50%",
                              background: "white",
                            }}
                          />
                        )}
                      </div>
                      <div>{datasetFile.name}</div>
                    </div>

                    {/* Delete (trash) button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setDatasetFile(null);
                        setDatasetName("");
                        localStorage.removeItem("customDataset");
                      }}
                      style={{
                        background: "rgba(255,0,0,0.08)",
                        border: "1px solid rgba(255,0,0,0.3)",
                        borderRadius: 6,
                        cursor: "pointer",
                        padding: 3,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                      title="Delete dataset"
                    >
                      <Trash2 size={15} color="rgb(255,80,80)" />
                    </button>
                  </div>
                ) : (
                  <div
                    style={{
                      opacity: 0.6,
                      fontSize: 13,
                      margin: "6px 0 10px 4px",
                    }}
                  >
                    No custom dataset uploaded
                  </div>
                )}

                {/* Divider */}
                <div
                  style={{
                    height: 1,
                    background: "rgba(255,255,255,0.12)",
                    margin: "6px 0",
                  }}
                />

                {/* Upload Button */}
                <button
                  onClick={() => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = ".csv,.zip";
                    input.onchange = (e) => {
                      const file = (e.target as HTMLInputElement)?.files?.[0];
                      if (file) {
                        setDatasetFile(file);
                        setDatasetName(file.name);
                        localStorage.setItem("customDataset", file.name);
                        setDatasetUploadedPopup({ name: file.name });
                      }
                    };
                    input.click();
                  }}
                  style={{
                    width: "100%",
                    padding: "10px 12px",
                    borderRadius: 10,
                    background: "rgba(255,255,255,0.08)",
                    border: "1px solid rgba(255,255,255,0.18)",
                    color: "white",
                    cursor: "pointer",
                    fontWeight: 600,
                  }}
                >
                  + Upload Dataset
                </button>
                <AnimatePresence>
                  {datasetUploadedPopup && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      style={{
                        position: "fixed",
                        inset: 0,
                        background: "rgba(0,0,0,0.45)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        zIndex: 2500,
                      }}
                      onClick={() => setDatasetUploadedPopup(null)}
                    >
                      <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.9, opacity: 0 }}
                        onClick={(e) => e.stopPropagation()}
                        style={{
                          width: 400,
                          borderRadius: 18,
                          padding: "26px 22px",
                          background: "rgba(30,30,35,0.95)",
                          border: "1px solid rgba(255,255,255,0.12)",
                          boxSizing: "border-box",
                          color: "white",
                          textAlign: "center",
                        }}
                      >
                        <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 14, color: "#7bff9d" }}>
                          Dataset Uploaded
                        </div>

                        <div
                          style={{
                            fontSize: 15,
                            opacity: 0.9,
                            marginBottom: 24,
                            wordWrap: "break-word",
                          }}
                        >
                          <span style={{ color: "#c8ffda" }}>{datasetUploadedPopup.name}</span> is now cached locally!
                        </div>

                        <button
                          onClick={() => setDatasetUploadedPopup(null)}
                          style={{
                            width: "100%",
                            padding: "10px 12px",
                            borderRadius: 10,
                            border: "1px solid rgba(120,255,150,0.4)",
                            background: "rgba(80,255,140,0.25)",
                            fontWeight: 600,
                            color: "white",
                            cursor: "pointer",
                          }}
                        >
                          OK
                        </button>
                      </motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
          </div>


          <Tooltip label="Trainer Config">
            <span>
              <IconButton onClick={() => setConfigOpen((v) => !v)}>
                <Settings size={22} />
              </IconButton>
            </span>
          </Tooltip>

          {/* Train button */}
          <Tooltip label="Train model">
            <span>
              <IconButton onClick={handleTrain}>
                <Activity size={24} />
              </IconButton>
            </span>
          </Tooltip>

          <Divider />

          {/* User menu */}
          <div style={{ position: "relative" }}>
            <button
              ref={userBtnRef}
              onClick={() => setUserOpen((v) => !v)}
              className="user-button"
              data-user-menu
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 10px",
                borderRadius: 10,
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.05)",
                cursor: "pointer",
              }}
              title={user?.email || ""}
            >
              <UserIcon size={20} />
              <MoreVertical size={18} />
            </button>

            {userOpen && (
              <div
                style={{
                  position: "absolute",
                  right: 0,
                  marginTop: 8,
                  minWidth: 240,
                  borderRadius: 12,
                  padding: 10,
                  background:
                    "linear-gradient(180deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.06) 100%)",
                  border: "1px solid rgba(255,255,255,0.15)",
                  backdropFilter: "blur(12px)",
                  boxShadow: "0 12px 28px rgba(0,0,0,0.35)",
                  zIndex: 600,
                }}
                data-user-menu
              >
                <div style={{ fontSize: 12, opacity: 0.8 }}>Signed in as</div>
                <div style={{ fontWeight: 700, marginBottom: 8 }}>{user?.email}</div>
                <button
                  onClick={() => {
                    logout();
                    navigate("/login");
                  }}
                  style={{
                    width: "100%",
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "10px 12px",
                    borderRadius: 10,
                    background: "transparent",
                    border: "1px solid rgba(255,255,255,0.12)",
                    cursor: "pointer",
                    color: "white",
                  }}
                >
                  <LogOut size={18} /> Logout
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* OVERLAY LAYOUT */}
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 0,
        }}
        onDragOver={onDragOver}
        onDrop={onDrop}
      >
        {loading ? (
          <div style={{ padding: 20, opacity: 0.75 }}>Loading project…</div>
        ) : (
          <ReactFlow
            proOptions={{ hideAttribution: true }}
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChanged}
            onEdgesChange={onEdgesChanged}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            onSelectionChange={onSelectionChange}
            style={{ width: "100%", height: "100%" }}
            panOnDrag={!locked}
            zoomOnScroll={!locked}
            zoomOnPinch={!locked}
            panOnScroll={!locked}
            nodesDraggable={!locked}
            nodesConnectable={!locked}
            elementsSelectable={!locked}
          >
            <Background />
            <Controls
              showZoom={false}
              showFitView={false}
              showInteractive={false}
              style={{
                position: "absolute",
                left: leftPanelWidth + 12,
                bottom: 20,
                zIndex: 500,
                display: "flex",
                flexDirection: "column",
                gap: "2px",
              }}
            >
              {/* ZOOM IN */}
              <button
                onClick={() => !locked && rf.zoomIn()}
                className="react-flow__controls-button"
                style={{ cursor: locked ? "not-allowed" : "pointer" }}
                title="Zoom In"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" stroke="black" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
              </button>

              {/* ZOOM OUT */}
              <button
                onClick={() => !locked && rf.zoomOut()}
                className="react-flow__controls-button"
                style={{ cursor: locked ? "not-allowed" : "pointer" }}
                title="Zoom Out"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" stroke="black" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
              </button>

              {/* LOCK */}
              <button
                onClick={() => setLocked((v) => !v)}
                className="react-flow__controls-button"
                style={{
                  opacity: locked ? 0.65 : 1,
                  cursor: "pointer",
                }}
                title={locked ? "Unlock Canvas" : "Lock Canvas"}
              >
                {locked ? (
                  <svg width="16" height="16" viewBox="0 0 24 24" stroke="black" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 24 24" stroke="black" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                    <path d="M7 11V7a5 5 0 0 1 9.9-1" />
                  </svg>
                )}
              </button>
            </Controls>
          </ReactFlow>
        )}
      </div>

      {/* LEFT SIDEBAR */}
      <aside
        className="left-sidebar"
        style={{
          position: "fixed",
          top: 56,
          left: 0,
          bottom: 0,
          width: leftPanelWidth,
          zIndex: 200,
          overflowY: "hidden",
          overflowX: "hidden",
          pointerEvents: leftCollapsed ? "none" : "auto",
          opacity: leftCollapsed ? 0.0 : 1,
          backdropFilter: "blur(6px)",
          background:
            "linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%)",
          borderRight: "1px solid rgba(255,255,255,0.08)",
          transition: "width .18s ease, opacity .12s ease",
          padding: 0,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Search */}
        <div style={{ padding: "10px 40px 10px 12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <LayersIcon size={16} opacity={0.85} />
            <div style={{ fontWeight: 700 }}>Layers</div>
          </div>
          <div style={{ marginTop: 8 }}>
            <input
              className="glass-input"
              placeholder="Search layers…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        </div>

        <div className="left-pane-scroll" style={{ overflowY: "auto", flex: 1 }}>
          {catalog === null ? (
            <div style={{ opacity: 0.7, padding: 8 }}>Loading…</div>
          ) : filteredCatalog.length === 0 ? (
            <div style={{ opacity: 0.7, padding: 8 }}>No layers found</div>
          ) : (
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {filteredCatalog.map((op) => (
              <li
                key={op.name}
                draggable
                onDragStart={(e) => e.dataTransfer.setData("application/layer", op.name)}
                style={{
                  margin: "4px 8px",
                  borderRadius: 10,
                  background: "rgba(255,255,255,0.04)",
                  cursor: "grab",
                  userSelect: "none",
                  lineHeight: "32px",
                  padding: "0 10px",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  transition: "background 0.15s ease, transform 0.15s ease",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget.style.background = "rgba(255,255,255,0.08)");
                  (e.currentTarget.style.transform = "translateY(-1px)");
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget.style.background = "rgba(255,255,255,0.04)");
                  (e.currentTarget.style.transform = "translateY(0)");
                }}
                title={op.name}
              >
                {op.name}
              </li>
            ))}
          </ul>
          )}
        </div>
      </aside>

      {/* FLOATING CIRCULAR TOGGLE */}
      <button
        className="bubble-toggle"
        data-collapsed={leftCollapsed}
        onClick={() => setLeftCollapsed(v => !v)}
        title={leftCollapsed ? "Show Layers" : "Hide Layers"}
        style={{
          top: bubbleTop,
          left: leftPanelWidth + bubbleGap, 
        }}
      />

      {/* RIGHT PROPERTIES PANEL */}
      <aside
        style={{
          position: "fixed",
          top: 56,
          right: 0,
          bottom: 0,
          width: "clamp(260px, 28vw, 380px)",
          zIndex: 200,
          backdropFilter: "blur(6px)",
          background:
            "linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%)",
          borderLeft: "1px solid rgba(255,255,255,0.08)",
          overflowY: "auto",
          padding: 12,
          transform: selectedNode ? "translateX(0)" : "translateX(100%)",
          transition: "transform .18s ease",
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Properties</div>

        {!selectedNode ? (
          <div style={{ opacity: 0.7 }}>Select a node</div>
        ) : selectedParams === null ? (
          <div style={{ opacity: 0.7 }}>Loading params…</div>
        ) : selectedParams.length === 0 ? (
          <div style={{ opacity: 0.7 }}>No params</div>
        ) : (
          <div style={{ display: "grid", gap: 10 }}>
            {selectedParams.map((p) => (
              <div
                key={p.name}
                style={{
                  borderRadius: 12,
                  padding: 12,
                  border: "1px solid rgba(255,255,255,0.14)",
                  background:
                    "linear-gradient(180deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.05) 100%)",
                }}
              >
                <div style={{ fontSize: 12, opacity: 0.85, marginBottom: 6 }}>{p.name}</div>
                <input
                  value={p.value ?? ""}
                  onChange={(e) => updateParam(p.name, e.target.value)}
                  placeholder="value"
                  style={{
                    width: "100%",
                    padding: "10px 12px",
                    boxSizing: "border-box",
                    borderRadius: 10,
                    border: "1px solid rgba(255,255,255,0.18)",
                    background: "rgba(0,0,0,0.25)",
                    color: "white",
                    outline: "none",
                  }}
                />
              </div>
            ))}
          </div>
        )}
      </aside>

      {/* TRAINER CONFIG PANEL */}
      <aside
        style={{
          position: "fixed",
          top: 56,
          right: 0,
          bottom: 0,
          width: "clamp(170px, 28vw, 100px)",
          zIndex: 210,
          backdropFilter: "blur(6px)",
          background:
            "linear-gradient(180deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%)",
          borderLeft: "1px solid rgba(255,255,255,0.1)",
          padding: 12,
          transform: configOpen ? "translateX(0)" : "translateX(100%)",
          transition: "transform .2s ease",
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Trainer Config</div>
        {["Epochs", "LearningRate", "BatchSize"].map((key) => (
          <div key={key} style={{ marginBottom: 8 }}>
            <label style={{ display: "block", fontSize: 12, marginBottom: 4 }}>{key}</label>
            <input
              type="number"
              value={trainerConfig[key as keyof typeof trainerConfig]}
              onChange={(e) =>
                setTrainerConfig((cfg) => ({
                  ...cfg,
                  [key]: parseFloat(e.target.value),
                }))
              }
              style={{
                width: "150px",
                padding: "8px 8px",
                borderRadius: 8,
                border: "1px solid rgba(255,255,255,0.18)",
                background: "rgba(0,0,0,0.25)",
                color: "white",
                textAlign: "center",
              }}
            />
          </div>
        ))}
      </aside>

      {/* Saving pulse indicator */}
      {saving && (
        <div
          style={{
            position: "fixed",
            bottom: 18,
            right: 18,
            padding: "10px 12px",
            borderRadius: 10,
            border: "1px solid rgba(120,200,255,0.35)",
            background: "rgba(40,80,120,0.3)",
            color: "white",
            animation: "pulseSave 1.4s ease-out infinite",
            backdropFilter: "blur(8px)",
            boxShadow: "0 8px 22px rgba(0,0,0,0.35)",
            zIndex: 9999,
          }}
        >
          Saving…
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div
          style={{
            position: "fixed",
            bottom: 18,
            left: "50%",
            transform: "translateX(-50%)",
            padding: "10px 12px",
            borderRadius: 10,
            border: "1px solid rgba(255,255,255,0.18)",
            background:
              "linear-gradient(180deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.06) 100%)",
            color: "white",
            backdropFilter: "blur(8px)",
            boxShadow: "0 8px 22px rgba(0,0,0,0.35)",
            zIndex: 9999,
          }}
        >
          {toast}
        </div>
        
      )}
      {/* TRAIN OUTPUT OVERLAY */}
      {trainOverlay && (
        <div
          style={{
            position: "fixed",
            bottom: 0,
            left: 0,
            right: 0,
            height: "50%",
            background: "rgba(15,15,15,0.92)",
            backdropFilter: "blur(8px)",
            borderTop: "1px solid rgba(255,255,255,0.15)",
            zIndex: 9999,
            display: "flex",
            flexDirection: "row",
          }}
        >
          {/* Left: Output console */}
          <div
            style={{
              flex: 1,
              padding: 20,
              overflowY: "auto",
              color: "white",
              fontFamily: "monospace",
              fontSize: 13,
              borderRight: "1px solid rgba(255,255,255,0.15)",
              position: "relative",
            }}
          >
            {/* Stop Training Button */}
            <button
              onClick={async () => {
                try {
                  await fetch("http://localhost:8000/cancel_training", { method: "POST" });
                  setAlertBox({ message: "Training Cancelled!", type: "red" });
                } catch (err: any) {
                    setAlertBox({ message: "Failed To Cancel Training!", type: "red" });
                  }
              }}
              style={{
                position: "absolute",
                top: 10,
                right: 45,
                background: "rgba(255, 60, 60, 0.2)",
                border: "1px solid rgba(255, 60, 60, 0.5)",
                borderRadius: 8,
                color: "rgb(255, 120, 120)",
                padding: "6px 10px",
                cursor: "pointer",
                fontWeight: 600,
              }}
              title="Stop training"
            >
              ❌
            </button>

            {/* Console Text */}
            <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
              {trainOutput || "Training in progress..."}
            </pre>
          </div>

          {/* Right: Graph Dashboard */}
          <div
            style={{
              flex: 1,
              padding: 20,
              color: "white",
              display: "flex",
              flexDirection: "column",
              overflowY: "auto",
            }}
          >
            {/* Tabs */}
            <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
              {(["Loss", "Speed", "Time/Batch", "ETA"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  style={{
                    padding: "6px 10px",
                    borderRadius: 6,
                    border: "1px solid rgba(255,255,255,0.25)",
                    background:
                      activeTab === tab ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.05)",
                    color: "white",
                    cursor: "pointer",
                    fontSize: 12,
                  }}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Graph Space */}
            <div
              style={{
                flex: 1,
                width: "100%",
                height: "100%",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              {renderMetricChart(activeTab)}
            </div>
          </div>

          {/* Close Button */}
          <button
            onClick={() => setTrainOverlay(false)}
            style={{
              position: "absolute",
              top: 10,
              right: 10,
              background: "rgba(255,255,255,0.1)",
              border: "1px solid rgba(255,255,255,0.25)",
              borderRadius: 8,
              color: "white",
              padding: "6px 10px",
              cursor: "pointer",
            }}
            title="Close output panel"
          >
            ✕
          </button>
        </div>
      )}
      {/* Global Alert Popup */}
      <AnimatePresence>
        {alertBox && (
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
              zIndex: 2500,
            }}
            onClick={() => setAlertBox(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              style={{
                width: 420,
                borderRadius: 18,
                padding: "26px 24px",
                background: "rgba(35,35,35,0.92)",
                border: "1px solid rgba(255,255,255,0.15)",
                boxSizing: "border-box",
                color: "white",
                textAlign: "center",
              }}
            >
              {/* Title */}
              <div
                style={{
                  fontSize: 18,
                  fontWeight: 700,
                  marginBottom: 16,
                  color: alertBox.type === "red" ? "rgb(255,90,90)" : "rgb(120,160,255)",
                }}
              >
                {alertBox.type === "red" ? "Alert" : "Notice"}
              </div>

              {/* Message */}
              <div
                style={{
                  fontSize: 15,
                  opacity: 0.9,
                  marginBottom: 22,
                  whiteSpace: "pre-wrap",
                }}
              >
                {alertBox.message}
              </div>

              {/* OK Button */}
              <button
                onClick={() => setAlertBox(null)}
                style={{
                  width: "100%",
                  padding: "10px 12px",
                  borderRadius: 10,
                  border:
                    alertBox.type === "red"
                      ? "1px solid rgba(255,80,80,0.4)"
                      : "1px solid rgba(120,180,255,0.4)",
                  background:
                    alertBox.type === "red"
                      ? "rgba(255,50,50,0.25)"
                      : "rgba(70,140,255,0.25)",
                  fontWeight: 600,
                  color: "white",
                  cursor: "pointer",
                }}
              >
                OK
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Builder() {
  return (
    <ReactFlowProvider>
      <BuilderInner />
    </ReactFlowProvider>
  );
}