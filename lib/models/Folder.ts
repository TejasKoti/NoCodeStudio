import mongoose from "mongoose";

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

export const Folder = mongoose.models.Folder || mongoose.model<IFolder>("Folder", FolderSchema);