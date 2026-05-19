import type { ForecastPayload, ForecastResponse } from './types';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api';

function apiUrl(path: string): string {
  const base = API_BASE_URL.replace(/\/$/, '');
  return `${base}${path}`;
}

export async function fetchHealth(): Promise<string> {
  const response = await fetch(apiUrl('/health'), { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  const data = (await response.json()) as { status?: string };
  return data.status ?? 'unknown';
}

export async function requestForecast(payload: ForecastPayload): Promise<ForecastResponse> {
  const response = await fetch(apiUrl('/forecast'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body);
    throw new Error(detail || `Forecast failed: ${response.status}`);
  }
  return body as ForecastResponse;
}
