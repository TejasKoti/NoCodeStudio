import mongoose from "mongoose";

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

export const Project =
  mongoose.models.Project || mongoose.model<IProject>("Project", ProjectSchema);