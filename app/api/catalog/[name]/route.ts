import { NextResponse } from "next/server";

export async function GET() {
  try {
    const res = await fetch(process.env.ML_CATALOG_URL!);

    if (!res.ok) {
      throw new Error(`Modal catalog failed: ${res.status}`);
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (e) {
    console.error("catalog error:", e);
    return NextResponse.json(
      { layers: [] },
      { status: 500 }
    );
  }
}