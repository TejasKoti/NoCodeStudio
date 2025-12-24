import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const res = await fetch(
      process.env.MODAL_CANCEL_TRAINING_URL!,
      {
        method: "POST",
      }
    );

    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json(
        { error: text || "Modal cancel failed" },
        { status: 500 }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message || "Cancel request failed" },
      { status: 500 }
    );
  }
}