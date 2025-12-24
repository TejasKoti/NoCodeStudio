import { NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import { requireUserId } from "@/lib/auth";
import { Project } from "@/lib/models/Project";
import { Folder } from "@/lib/models/Folder";

/* =========================
   GET /api/projects/[id]
========================= */
export async function GET(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    await connectDB();
    const userId = requireUserId(req);
    const { id } = await params;

    const project = await Project.findOne({
      _id: id,
      userId,
    });

    if (!project) {
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }

    return NextResponse.json({ project });
  } catch (err: any) {
    const status = err?.status || 500;
    return NextResponse.json(
      {
        message:
          status === 401
            ? "Unauthorized"
            : "Server error fetching project",
      },
      { status }
    );
  }
}

/* =========================
   PUT /api/projects/[id]
========================= */
export async function PUT(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    await connectDB();
    const userId = requireUserId(req);
    const { id } = await params;

    const body = await req.json();
    const { title, name, description, tags, thumbnail, graph } = body;

    const update: Record<string, any> = {};
    if (title || name) update.title = (title ?? name).trim();
    if (typeof description === "string") update.description = description;
    if (Array.isArray(tags)) update.tags = tags;
    if (thumbnail !== undefined) update.thumbnail = thumbnail;
    if (graph !== undefined) update.graph = graph;

    const project = await Project.findOneAndUpdate(
      { _id: id, userId },
      update,
      { new: true }
    );

    if (!project) {
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }

    return NextResponse.json({ project });
  } catch (err: any) {
    const status = err?.status || 500;
    return NextResponse.json(
      {
        message:
          status === 401
            ? "Unauthorized"
            : "Server error updating project",
      },
      { status }
    );
  }
}

/* =========================
   DELETE /api/projects/[id]
========================= */
export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    await connectDB();
    const userId = requireUserId(req);
    const { id } = await params;

    const project = await Project.findOneAndDelete({
      _id: id,
      userId,
    });

    if (!project) {
      return NextResponse.json(
        { message: "Project not found" },
        { status: 404 }
      );
    }

    await Folder.updateMany(
      { projects: id },
      { $pull: { projects: id } }
    );

    return NextResponse.json({
      message: "Project deleted successfully",
    });
  } catch (err: any) {
    const status = err?.status || 500;
    return NextResponse.json(
      {
        message:
          status === 401
            ? "Unauthorized"
            : "Server error deleting project",
      },
      { status }
    );
  }
}