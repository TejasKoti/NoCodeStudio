import { NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import { requireUserId } from "@/lib/auth";
import { Project } from "@/lib/models/Project";
import { Folder } from "@/lib/models/Folder";

export async function GET(req: Request) {
  try {
    await connectDB();
    const userId = requireUserId(req);
    const projects = await Project.find({ userId }).sort({ updatedAt: -1 });
    return NextResponse.json({ projects });
  } catch (err: any) {
    const status = err?.status || 500;
    return NextResponse.json({ message: status === 401 ? "Unauthorized" : "Server error fetching projects" }, { status });
  }
}

export async function POST(req: Request) {
  try {
    await connectDB();
    const userId = requireUserId(req);

    const body = await req.json();
    const { title, name, description, tags, thumbnail, graph, folderId } = body;

    const finalTitle = ((title ?? name ?? "Untitled Project") as string).trim() || "Untitled Project";

    const project = await Project.create({
      userId,
      folderId: folderId ?? null,
      title: finalTitle,
      description: description ?? "",
      tags: Array.isArray(tags) ? tags : [],
      thumbnail: thumbnail ?? null,
      graph,
    });

    if (folderId) {
      await Folder.findByIdAndUpdate(folderId, { $addToSet: { projects: project._id } });
    }

    return NextResponse.json({ project }, { status: 201 });
  } catch (err: any) {
    const status = err?.status || 400;
    return NextResponse.json({ error: status === 401 ? "Unauthorized" : "Unable to create project" }, { status });
  }
}