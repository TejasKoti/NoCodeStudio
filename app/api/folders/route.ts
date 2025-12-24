import { NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import { requireUserId } from "@/lib/auth";
import { Folder } from "@/lib/models/Folder";

export async function GET(req: Request) {
  try {
    await connectDB();
    const userId = requireUserId(req);

    const folders = await Folder.find({ userId }).sort({ createdAt: 1 });

    return NextResponse.json({ folders });
  } catch (err: any) {
    const status = err?.status || 500;
    return NextResponse.json(
      { message: status === 401 ? "Unauthorized" : "Server error fetching folders" },
      { status }
    );
  }
}

export async function POST(req: Request) {
  try {
    await connectDB();
    const userId = requireUserId(req);

    const body = await req.json();
    const name = (body?.name || "").trim();
    if (!name) {
      return NextResponse.json({ error: "Folder name required" }, { status: 400 });
    }

    const folder = await Folder.create({
      userId,
      name,
      projects: [],
    });

    return NextResponse.json({ folder }, { status: 201 });
  } catch (err: any) {
    const status = err?.status || 500;
    return NextResponse.json(
      { error: status === 401 ? "Unauthorized" : "Unable to create folder" },
      { status }
    );
  }
}