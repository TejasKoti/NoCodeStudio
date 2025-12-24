import { NextResponse } from "next/server";
import { mlTrain } from "@/lib/python";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const data = await mlTrain(body);
    return NextResponse.json(data);
  } catch (e: any) {
    console.error("train error:", e);
    return NextResponse.json(
      {
        error: "Failed to train model",
        detail: e?.message ?? String(e),
      },
      { status: 500 }
    );
  }
}