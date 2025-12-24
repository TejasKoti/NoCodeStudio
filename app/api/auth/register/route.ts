import { NextResponse } from "next/server";
import { connectDB } from "@/lib/db";
import { User } from "@/lib/models/User";
import { signToken } from "@/lib/auth";

export async function POST(req: Request) {
  try {
    await connectDB();
    const { email, password } = await req.json();

    if (!email || !password) {
      return NextResponse.json({ message: "Email and password are required" }, { status: 400 });
    }

    const existing = await User.findOne({ email });
    if (existing) {
      return NextResponse.json({ message: "User already exists" }, { status: 400 });
    }

    const user = await User.create({ email, password });
    const token = signToken(user._id.toString());

    return NextResponse.json(
      { token, user: { id: user._id, email: user.email } },
      { status: 201 }
    );
  } catch (err) {
    console.error("REGISTER ERROR", err);
    return NextResponse.json({ message: "Server error registering user" }, { status: 500 });
  }
}