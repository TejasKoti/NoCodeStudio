import { NextResponse } from "next/server";
import { mlImport } from "@/lib/python";

export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const file = form.get("file");

    if (!file || !(file instanceof File)) {
      return NextResponse.json(
        { error: "No file uploaded" },
        { status: 400 }
      );
    }
    const code = await file.text();
    const data = await mlImport(code);

    return NextResponse.json(data);
  } catch (e) {
    console.error("import error:", e);
    return NextResponse.json(
      { error: "Import failed", detail: String(e) },
      { status: 500 }
    );
  }
}