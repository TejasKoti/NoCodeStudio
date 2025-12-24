import { NextResponse } from "next/server";
import { mlLayer } from "@/lib/python";

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const name = searchParams.get("name");

    if (!name) {
      return NextResponse.json(
        { error: "Missing layer name", params: [] },
        { status: 400 }
      );
    }

    const data = await mlLayer(name);
    return NextResponse.json(data);
  } catch (e) {
    console.error("layer error:", e);
    return NextResponse.json({ params: [] }, { status: 500 });
  }
}