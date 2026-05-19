export type CsvRow = Record<string, string | number | null>;

export type BackendName = 'seasonal_naive' | 'chronos2' | 'timesfm';

export interface ForecastPayload {
  records: CsvRow[];
  future_records?: CsvRow[];
  prediction_length: number;
  backend: BackendName;
  target: string;
  device?: string;
  model_id?: string;
}

export interface ForecastResponse {
  rows: CsvRow[];
}
