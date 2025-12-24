import { NextResponse } from "next/server";

const PY = process.env.PY_SERVICE_URL || "http://localhost:8000";

export async function GET() {
  try {
    const r = await fetch(`${PY}/health`);
    const data = await r.json().catch(() => ({}));
    return NextResponse.json({ python: data?.status ?? "ok" });
  } catch {
    return NextResponse.json({ python: "unreachable" }, { status: 503 });
  }
}