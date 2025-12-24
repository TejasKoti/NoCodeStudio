import mongoose from "mongoose";

type MongooseCache = {
  conn: typeof mongoose | null;
  promise: Promise<typeof mongoose> | null;
};

declare global {
  var __mongooseCache: MongooseCache | undefined;
}

const cache: MongooseCache =
  global.__mongooseCache ?? { conn: null, promise: null };

global.__mongooseCache = cache;

function getMongoUri(): string {
  const uri = process.env.MONGO_URI;
  if (!uri) {
    throw new Error("MONGO_URI is not set");
  }
  return uri;
}

export async function connectDB() {
  if (cache.conn) return cache.conn;

  if (!cache.promise) {
    cache.promise = mongoose
      .connect(getMongoUri(), {
        family: 4,
      })
      .then((m) => m);
  }

  cache.conn = await cache.promise;
  return cache.conn;
}