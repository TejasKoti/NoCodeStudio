import { NextResponse } from "next/server";
import { mlExport } from "@/lib/python";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const data = await mlExport(body);
    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json(
      { error: "export failed", detail: String(e) },
      { status: 500 }
    );
  }
}