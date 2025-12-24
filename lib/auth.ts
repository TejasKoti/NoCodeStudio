import jwt from "jsonwebtoken";

function getJwtSecret(): string {
  const secret = process.env.JWT_SECRET;
  if (!secret) {
    throw new Error("JWT_SECRET is not set");
  }
  return secret;
}

export function signToken(userId: string) {
  return jwt.sign(
    { id: userId },
    getJwtSecret(),
    { expiresIn: "7d" }
  );
}

export function getUserIdFromAuthHeader(req: Request): string | null {
  const authHeader = req.headers.get("authorization") || "";
  if (!authHeader.startsWith("Bearer ")) return null;

  const token = authHeader.split(" ")[1];

  try {
    const decoded = jwt.verify(
      token,
      getJwtSecret()
    ) as { id: string };

    return decoded.id;
  } catch {
    return null;
  }
}

export function requireUserId(req: Request): string {
  const userId = getUserIdFromAuthHeader(req);
  if (!userId) {
    const err = new Error("Unauthorized");
    // @ts-ignore
    err.status = 401;
    throw err;
  }
  return userId;
}