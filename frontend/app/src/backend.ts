export type BackendName = "seasonal_naive" | "chronos2" | "timesfm";

export type CsvCell = string | number | null;
export type CsvRow = Record<string, CsvCell>;

export interface ForecastPoint {
  label: string;
  value: number;
}

export interface LoadedData {
  headers: string[];
  rows: CsvRow[];
  datetimeKey: string | null;
}

interface ForecastPayload {
  records: CsvRow[];
  future_records?: CsvRow[];
  prediction_length: number;
  backend: BackendName;
  target: string;
}

interface ForecastResponse {
  rows: CsvRow[];
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

function apiUrl(path: string): string {
  return `${String(API_BASE_URL).replace(/\/$/, "")}${path}`;
}

export async function fetchHealth(): Promise<string> {
  const response = await fetch(apiUrl("/health"), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  const body = (await response.json()) as { status?: string };
  return body.status ?? "unknown";
}

export async function requestForecast(payload: ForecastPayload): Promise<ForecastResponse> {
  const response = await fetch(apiUrl("/forecast"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body);
    throw new Error(detail || `Forecast failed: ${response.status}`);
  }
  return body as ForecastResponse;
}

export async function loadFromCSV(text: string): Promise<LoadedData> {
  const rows = parseCSV(text);
  const headers = Object.keys(rows[0] ?? {});
  return {
    headers,
    rows,
    datetimeKey: guessDatetimeKey(headers),
  };
}

export async function loadFromXLSX(buffer: ArrayBuffer): Promise<LoadedData> {
  const XLSX = await import("xlsx");
  const workbook = XLSX.read(buffer, { type: "array" });
  const sheetName = workbook.SheetNames[0];
  if (!sheetName) {
    throw new Error("XLSX workbook does not contain a sheet");
  }
  const sheet = workbook.Sheets[sheetName];
  const rows = XLSX.utils.sheet_to_json<Record<string, unknown>>(sheet, { defval: null });
  const normalizedRows = rows.map((row) =>
    Object.fromEntries(Object.entries(row).map(([key, value]) => [key, normalizeCell(value)]))
  );
  const headers = Object.keys(normalizedRows[0] ?? {});
  return {
    headers,
    rows: normalizedRows,
    datetimeKey: guessDatetimeKey(headers),
  };
}

export function rowsToForecastPoints(rows: CsvRow[], target: string): ForecastPoint[] {
  return rows
    .map((row, index) => {
      const value = firstNumeric(row[target], row["predictions"], row["0.5"], row["mean"], row["median"]);
      if (value === null) return null;
      return {
        label: String(row.timestamp ?? row.date ?? row.datetime ?? row.ds ?? index),
        value,
      };
    })
    .filter((point): point is ForecastPoint => point !== null);
}

function parseCSV(text: string): CsvRow[] {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) return [];

  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, parseValue(values[index] ?? "")])) as CsvRow;
  });
}

function splitCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"' && line[i + 1] === '"') {
      current += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  result.push(current.trim());
  return result;
}

function parseValue(value: string): CsvCell {
  if (value === "") return null;
  const numeric = Number(value);
  if (Number.isFinite(numeric)) return numeric;
  return value;
}

function normalizeCell(value: unknown): CsvCell {
  if (value == null) return null;
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return String(value);
}

function guessDatetimeKey(headers: string[]): string | null {
  const lowered = headers.map((header) => header.toLowerCase());
  const candidates = ["timestamp", "datetime", "date", "time", "ds"];
  const index = lowered.findIndex((header) => candidates.includes(header));
  return index >= 0 ? headers[index] : headers[0] ?? null;
}

function firstNumeric(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return null;
}
