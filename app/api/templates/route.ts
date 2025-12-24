import { NextResponse } from "next/server";

export async function GET() {
  const templates = [
    { _id: "1", name: "Classification" },
    { _id: "2", name: "Segmentation" },
    { _id: "3", name: "Object Detection" },
  ];
  return NextResponse.json({ templates });
}