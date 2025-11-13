import mongoose from "mongoose";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import axios from "axios";
import "dotenv/config";
import express, { Request, Response, NextFunction } from "express";
import cors from "cors";
import { Router } from "express";

/*
User Model
*/
export interface IUser extends mongoose.Document {
  email: string;
  password: string;
  comparePassword: (candidate: string) => Promise<boolean>;
}

const UserSchema = new mongoose.Schema<IUser>(
  {
    email: { type: String, required: true, unique: true, lowercase: true, trim: true },
    password: { type: String, required: true, minlength: 6 },
  },
  { timestamps: true }
);

UserSchema.pre("save", async function (next) {
  const user = this as IUser;
  if (!user.isModified("password")) return next();
  const salt = await bcrypt.genSalt(10);
  user.password = await bcrypt.hash(user.password, salt);
  next();
});

UserSchema.methods.comparePassword = function (candidate: string) {
  return bcrypt.compare(candidate, this.password);
};

export const User = mongoose.model<IUser>("User", UserSchema);

/*
Project Model
*/
const GraphSchema = new mongoose.Schema(
  {
    nodes: { type: Array, default: [] },
    edges: { type: Array, default: [] },
  },
  { _id: false }
);

export interface IProject extends mongoose.Document {
  userId: mongoose.Types.ObjectId;
  folderId?: mongoose.Types.ObjectId | null;
  title: string;
  description?: string;
  tags?: string[];
  thumbnail?: string | null;
  graph: { nodes: any[]; edges: any[] };
  createdAt: Date;
  updatedAt: Date;
}

const ProjectSchema = new mongoose.Schema<IProject>(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
    folderId: { type: mongoose.Schema.Types.ObjectId, ref: "Folder", default: null },
    title: { type: String, required: true },
    description: { type: String, default: "" },
    tags: { type: [String], default: [] },
    thumbnail: { type: String, default: null },
    graph: { type: GraphSchema, default: { nodes: [], edges: [] } },
  },
  { timestamps: true }
);

export const Project = mongoose.model<IProject>("Project", ProjectSchema);

/*
Folder Model
*/
export interface IFolder extends mongoose.Document {
  userId: mongoose.Types.ObjectId;
  name: string;
  projects: mongoose.Types.ObjectId[];
  createdAt: Date;
  updatedAt: Date;
}

const FolderSchema = new mongoose.Schema<IFolder>(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
    name: { type: String, required: true },
    projects: [{ type: mongoose.Schema.Types.ObjectId, ref: "Project", default: [] }],
  },
  { timestamps: true }
);

export const Folder = mongoose.model<IFolder>("Folder", FolderSchema);

/*
Authentication Middleware
*/
export interface AuthRequest extends Request {
  user?: { id: string };
}

export const authMiddleware = (req: AuthRequest, res: Response, next: NextFunction) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ error: "No token provided" });
  }
  const token = authHeader.split(" ")[1];

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET as string) as { id: string };
    req.user = { id: decoded.id };
    next();
  } catch (err) {
    console.error("JWT Error:", err);
    return res.status(401).json({ error: "Invalid token" });
  }
};

declare global {
  namespace Express {
    interface Request {
      user?: { id: string };
    }
  }
}

/*
Authentication Routes
*/
const authRouter = Router();

authRouter.post("/register", async (req: Request, res: Response) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ message: "Email and password are required" });

    const existing = await User.findOne({ email });
    if (existing)
      return res.status(400).json({ message: "User already exists" });

    const user = await User.create({ email, password });
    const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET as string, {
      expiresIn: "7d",
    });

    return res.status(201).json({ token, user: { id: user._id, email: user.email } });
  } catch (err) {
    console.error("REGISTER ERROR", err);
    return res.status(500).json({ message: "Server error registering user" });
  }
});

authRouter.post("/login", async (req: Request, res: Response) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ message: "Email and password required" });

    const user = await User.findOne({ email });
    if (!user)
      return res.status(400).json({ message: "Invalid credentials" });

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch)
      return res.status(400).json({ message: "Invalid credentials" });

    const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET as string, {
      expiresIn: "7d",
    });

    return res.json({ token, user: { id: user._id, email: user.email } });
  } catch (err) {
    console.error("LOGIN ERROR", err);
    return res.status(500).json({ message: "Server error logging in" });
  }
});

/*
Project Routes
*/
import { Router as Router2 } from "express";
interface AuthedRequest extends Request {
  user?: { id: string };
}
const projectRouter = Router2();

projectRouter.post("/", authMiddleware, async (req: AuthedRequest, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    const { title, name, description, tags, thumbnail, graph, folderId } = req.body;
    const finalTitle = (title ?? name ?? "Untitled Project").trim() || "Untitled Project";

    const project = await Project.create({
      userId,
      folderId: folderId ?? null,
      title: finalTitle,
      description: description ?? "",
      tags: Array.isArray(tags) ? tags : [],
      thumbnail: thumbnail ?? null,
      graph,
    });

    // Add project to folder if folderId exists
    if (folderId) {
      await Folder.findByIdAndUpdate(folderId, {
        $addToSet: { projects: project._id },
      });
    }

    return res.status(201).json({ project });
  } catch (err) {
    console.error("POST /projects ERROR", err);
    return res.status(400).json({ error: "Unable to create project" });
  }
});

projectRouter.get("/", authMiddleware, async (req: AuthedRequest, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) return res.status(401).json({ error: "Unauthorized" });
    const projects = await Project.find({ userId }).sort({ updatedAt: -1 });
    res.json({ projects });
  } catch (err) {
    console.error("GET /projects ERROR", err);
    res.status(500).json({ message: "Server error fetching projects" });
  }
});

projectRouter.get("/:id", authMiddleware, async (req: AuthedRequest, res: Response) => {
  try {
    const userId = req.user?.id;
    const { id } = req.params;
    const project = await Project.findOne({ _id: id, userId });
    if (!project) return res.status(404).json({ error: "Not found" });
    res.json({ project });
  } catch (err) {
    console.error("GET /projects/:id ERROR", err);
    res.status(500).json({ message: "Server error fetching project" });
  }
});

projectRouter.put("/:id", authMiddleware, async (req: AuthedRequest, res: Response) => {
  try {
    const userId = req.user?.id;
    const { id } = req.params;
    const { title, name, description, tags, thumbnail, graph } = req.body;

    const update: Record<string, any> = {};
    if (title || name) update.title = (title ?? name).trim();
    if (typeof description === "string") update.description = description;
    if (Array.isArray(tags)) update.tags = tags;
    if (typeof thumbnail === "string" || thumbnail === null) update.thumbnail = thumbnail;
    if (graph !== undefined) update.graph = graph;

    const project = await Project.findOneAndUpdate({ _id: id, userId }, update, { new: true });
    if (!project) return res.status(404).json({ error: "Not found" });

    res.json({ project });
  } catch (err) {
    console.error("PUT /projects/:id ERROR", err);
    res.status(500).json({ message: "Server error updating project" });
  }
});

/*
Folder Routes
*/
import { Router as Router3 } from "express";
const folderRouter = Router3();

folderRouter.post("/", authMiddleware, async (req: AuthedRequest, res: Response) => {
  try {
    const userId = req.user?.id;
    const { name } = req.body;
    if (!name || !name.trim()) return res.status(400).json({ message: "Folder name required" });

    const folder = await Folder.create({ userId, name: name.trim() });
    res.status(201).json({ folder });
  } catch (err) {
    console.error("POST /folders ERROR", err);
    res.status(500).json({ error: "Unable to create folder" });
  }
});

folderRouter.get("/", authMiddleware, async (req: AuthedRequest, res: Response) => {
  try {
    const userId = req.user?.id;
    const folders = await Folder.find({ userId }).sort({ updatedAt: -1 });
    res.json({ folders });
  } catch (err) {
    console.error("GET /folders ERROR", err);
    res.status(500).json({ message: "Server error fetching folders" });
  }
});

folderRouter.delete("/:id", authMiddleware, async (req: AuthedRequest, res: Response) => {
  try {
    const userId = req.user?.id;
    const { id } = req.params;
    const folder = await Folder.findOneAndDelete({ _id: id, userId });
    if (!folder) return res.status(404).json({ message: "Folder not found" });
    await Project.updateMany({ _id: { $in: folder.projects } }, { $unset: { folderId: "" } });
    res.json({ message: "Folder deleted" });
  } catch (err) {
    console.error("DELETE /folders/:id ERROR", err);
    res.status(500).json({ message: "Server error deleting folder" });
  }
});

folderRouter.put("/:id/projects", authMiddleware, async (req: AuthedRequest, res: Response) => {
  try {
    const userId = req.user?.id;
    const { id } = req.params;
    const { projectId, action } = req.body;

    const folder = await Folder.findOne({ _id: id, userId });
    if (!folder) return res.status(404).json({ message: "Folder not found" });

    if (action === "add") {
      if (!folder.projects.includes(projectId)) folder.projects.push(projectId);
    } else if (action === "remove") {
      folder.projects = folder.projects.filter((p) => p.toString() !== projectId);
    }

    await folder.save();
    res.json({ folder });
  } catch (err) {
    console.error("PUT /folders/:id/projects ERROR", err);
    res.status(500).json({ message: "Server error updating folder" });
  }
});

// Delete Project
projectRouter.delete("/:id", authMiddleware, async (req: AuthedRequest, res: Response) => {
  try {
    const userId = req.user?.id;
    const { id } = req.params;
    const project = await Project.findOneAndDelete({ _id: id, userId });
    if (!project) return res.status(404).json({ message: "Project not found" });
    await Folder.updateMany({ projects: id }, { $pull: { projects: id } });
    return res.json({ message: "Project deleted successfully" });
  } catch (err) {
    console.error("DELETE /projects/:id ERROR", err);
    return res.status(500).json({ message: "Server error deleting project" });
  }
});


/*
Server Bootstrap
*/
const app = express();
app.use(
  cors({
    origin: "http://localhost:5173",
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);
app.use(express.json());

import { Router as ExpressGroupRouter } from "express";
const apiRouter = ExpressGroupRouter();
apiRouter.use("/auth", authRouter);
apiRouter.use("/projects", projectRouter);
apiRouter.use("/folders", folderRouter);

apiRouter.get("/catalog", async (_req, res) => {
  try {
    const r = await axios.get("http://localhost:8000/catalog");
    return res.json(r.data);
  } catch (e) {
    console.error("catalog error:", e);
    return res.status(500).json({ layers: [] });
  }
});

// Train Route
apiRouter.post("/train", async (req, res) => {
  try {
    const r = await axios.post(`${process.env.PY_SERVICE_URL}/api/train`, req.body);
    return res.json(r.data);
  } catch (e: any) {
    console.error("train error:", e.response?.data || e.message);
    return res.status(500).json({
      error: e.response?.data?.error || "Failed to train model",
      stdout: e.response?.data?.stdout || "",
    });
  }
});

apiRouter.get("/layer/:name", async (req, res) => {
  try {
    const { name } = req.params;
    const r = await axios.get(`http://localhost:8000/layer/${name}`);
    return res.json(r.data);
  } catch (e) {
    console.error("layer error:", e);
    return res.status(500).json({ params: [] });
  }
});

// Templates Route
apiRouter.get("/templates", async (_req, res) => {
  try {
    // Concept Pladeholder lmao (Needs update to add playable templates)
    const templates = [
      { _id: "1", name: "Classification" },
      { _id: "2", name: "Segmentation" },
      { _id: "3", name: "Object Detection" },
    ];
    res.json({ templates });
  } catch (e) {
    console.error("templates error:", e);
    res.status(500).json({ templates: [] });
  }
});

// Import Route
import fileUpload from "express-fileupload";
import type { UploadedFile } from "express-fileupload";
import FormData from "form-data";
import fs from "fs";

app.use(fileUpload({ useTempFiles: true }));

apiRouter.post("/import", async (req, res) => {
  try {
    const file = (req.files?.file as UploadedFile) || null;
    if (!file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const formData = new FormData();
    formData.append("file", fs.createReadStream(file.tempFilePath), file.name);

    const response = await axios.post("http://localhost:8000/import", formData, {
      headers: formData.getHeaders(),
    });

    return res.json(response.data);
  } catch (e) {
    console.error("import error:", e);
    return res.status(500).json({ error: "Import failed" });
  }
});

// Export Route
apiRouter.post("/export", async (req, res) => {
  try {
    const response = await axios.post("http://localhost:8000/export", req.body, {
      headers: { "Content-Type": "application/json" },
    });

    return res.json(response.data);
  } catch (e) {
    console.error("export error:", e);
    return res.status(500).json({ error: "Export failed" });
  }
});

// Run Route
apiRouter.post("/run", async (req, res) => {
  try {
    const response = await axios.post(`${process.env.PY_SERVICE_URL}/api/run`, req.body, {
      headers: { "Content-Type": "application/json" },
    });
    return res.json(response.data);
  } catch (err: any) {
    console.error("Python /api/run failed:", err?.response?.data || err.message);
    return res.status(err?.response?.status || 500).json({
      error: err?.response?.data?.error || "Run failed",
      stdout: err?.response?.data?.stdout || "",
      detail: err?.response?.data || err.message,
    });
  }
});

/*
Final Server Bootstrap Segment
*/

app.use("/api", apiRouter);

// Root route for quick sanity check
app.get("/", (_req, res) => res.json({ message: "API OK" }));

// Read env vars
const MONGO_URI = process.env.MONGO_URI;
const PORT = process.env.PORT || 3001;
const PY_SERVICE_URL = process.env.PY_SERVICE_URL || "http://localhost:8000";

if (!MONGO_URI) {
  throw new Error("MONGO_URI not found in .env");
}

// Connect to MongoDB
(async () => {
  try {
    await mongoose.connect(MONGO_URI);
    console.log("MongoDB connected");
  } catch (err) {
    console.error("MongoDB connection failed:", err);
    process.exit(1);
  }
})();

// Health check for Python service
app.get("/api/health/python", async (_req, res) => {
  try {
    const r = await axios.get(`${PY_SERVICE_URL}/health`);
    res.json({ python: r.data.status });
  } catch {
    res.status(503).json({ python: "unreachable" });
  }
});

// Start Express server
app.listen(PORT, () =>
  console.log(`API server running on http://localhost:${PORT}`)
);