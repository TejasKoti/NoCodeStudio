import { NextResponse } from "next/server";
import { mlRun } from "@/lib/python";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const data = await mlRun(body);
    return NextResponse.json(data);
  } catch (e: any) {
    console.error("run error:", e);
    return NextResponse.json(
      { error: "Run failed", detail: e?.message ?? String(e) },
      { status: 500 }
    );
  }
}