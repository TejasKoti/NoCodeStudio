function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing env var: ${name}`);
  return v;
}

const ML_CATALOG_URL = requireEnv("ML_CATALOG_URL");
const ML_LAYER_URL = requireEnv("ML_LAYER_URL");
const ML_EXPORT_URL = requireEnv("ML_EXPORT_URL");
const ML_IMPORT_URL = requireEnv("ML_IMPORT_URL");
const ML_RUN_URL = requireEnv("ML_RUN_URL");
const ML_TRAIN_URL = requireEnv("ML_TRAIN_URL");

async function handle(res: Response, label: string) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data?.error || `${label} failed (${res.status})`);
  }
  return data;
}

/* ---------- READ ---------- */

export async function mlCatalog() {
  return handle(await fetch(ML_CATALOG_URL), "catalog");
}

export async function mlLayer(name: string) {
  return handle(
    await fetch(`${ML_LAYER_URL}?name=${encodeURIComponent(name)}`),
    "layer"
  );
}

/* ---------- EXEC ---------- */

export async function mlExport(body: any) {
  return handle(
    await fetch(ML_EXPORT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
    "export"
  );
}

export async function mlImport(code: string) {
  return handle(
    await fetch(ML_IMPORT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    }),
    "import"
  );
}

export async function mlRun(body: any) {
  return handle(
    await fetch(ML_RUN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
    "run"
  );
}

export async function mlTrain(body: any) {
  return handle(
    await fetch(ML_TRAIN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
    "train"
  );
}