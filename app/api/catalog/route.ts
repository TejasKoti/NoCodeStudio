import { NextResponse } from "next/server";
import { mlCatalog } from "@/lib/python";

export async function GET() {
  try {
    const data = await mlCatalog();
    return NextResponse.json(data);
  } catch (e: any) {
    console.error("catalog error:", e);
    return NextResponse.json(
      { error: "Failed to load catalog", detail: String(e?.message || e) },
      { status: 500 }
    );
  }
}