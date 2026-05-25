import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  Pressable,
  ScrollView,
} from "react-native-web";
import type { BackendName, ForecastPoint, LoadedData } from "./backend";
import {
  fetchHealth,
  loadFromCSV,
  loadFromXLSX,
  requestForecast,
  rowsToForecastPoints,
} from "./backend";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Legend,
  Tooltip,
} from "chart.js";
import type { ChartData, ChartOptions } from "chart.js";

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Legend,
  Tooltip
);


type LineDataset = ChartData<"line">["datasets"][number];
type ModelKind = BackendName;


export default function App() {
  const [status, setStatus] = useState("idle");
  const [data, setData] = useState<LoadedData | null>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [modelKind, setModelKind] = useState<ModelKind>("seasonal_naive");
  const [model, setModel] = useState<{ backend: ModelKind } | null>(null);
  const [forecast, setForecast] = useState<string>("");
  const [forecastPoints, setForecastPoints] = useState<ForecastPoint[]>([]);

  useEffect(() => {
    let cancelled = false;

    async function loadSampleData(): Promise<void> {
      setStatus("loading sample data ...");

      try {
        const response = await fetch(
          `${import.meta.env.BASE_URL}data/sample_data.csv`,
          { cache: "no-store" }
        );

        if (!response.ok) {
          throw new Error(`failed to load data/sample_data.csv (${response.status})`);
        }

        const text = await response.text();
        const loaded = await loadFromCSV(text);

        if (cancelled) return;

        setData(loaded);
        setTarget(guessTarget(loaded));
        setModel(null);
        setForecast("");
        setForecastPoints([]);
        setStatus("sample data loaded");
      } catch (err: any) {
        if (cancelled) return;
        setStatus(`error: ${err.message || String(err)}`);
      }
    }

    void loadSampleData();

    return () => {
      cancelled = true;
    };
  }, []);

  async function handleFileChange(
    e: React.ChangeEvent<HTMLInputElement>
  ): Promise<void> {
    const file = e.target.files?.[0];
    if (!file) return;

    setStatus(`reading ${file.name} ...`);

    const ext = file.name.toLowerCase().split(".").pop();
    try {
      if (ext === "csv") {
        const text = await file.text();
        const loaded = await loadFromCSV(text);
        setData(loaded);
        setTarget(guessTarget(loaded));
        setStatus("data loaded");
      } else if (ext === "xlsx") {
        const buf = await file.arrayBuffer();
        const loaded = await loadFromXLSX(buf);
        setData(loaded);
        setTarget(guessTarget(loaded));
        setStatus("data loaded");
      } else {
        throw new Error("Unsupported file type (use .csv or .xlsx)");
      }

      setModel(null);
      setForecast("");
      setForecastPoints([]);
    } catch (err: any) {
      setStatus(`error: ${err.message || String(err)}`);
    }
  }

  function guessTarget(loaded: LoadedData): string | null {
    return loaded.headers.find((h) => h !== loaded.datetimeKey && h !== "id") ?? null;
  }

  function rowsForBackend(loaded: LoadedData, selectedTarget: string) {
    const timestampKey = loaded.datetimeKey ?? "timestamp";
    return loaded.rows.map((row) => ({
      id: String(row.id ?? "series_1"),
      timestamp: row[timestampKey],
      target: row[selectedTarget],
    }));
  }

  async function handleTrain(): Promise<void> {
    if (!data || !target) return;
    setStatus("training ...");
    try {
      await fetchHealth();
      setModel({ backend: modelKind });
      setForecast("");
      setForecastPoints([]);
      setStatus(`${modelKind} ready`);
    } catch (err: any) {
      setStatus(`error: ${err.message || String(err)}`);
    }
  }

  async function handlePredict(): Promise<void> {
    if (!data || !target || !model) return;

    setStatus("predicting 10 steps ...");

    try {
      const apiRows = rowsForBackend(data, target);
      const result = await requestForecast({
        records: apiRows,
        prediction_length: 10,
        backend: model.backend,
        target: "target",
      });
      const points = rowsToForecastPoints(result.rows, "target").slice(0, 10);
      setForecastPoints(points);

      setForecast(
        [
          `Model="${model.backend}" Target="${target}" → next 10 forecasts:`,
          ...points.map(
            (point, index) =>
              `+${index + 1} ${point.label}: ${point.value.toFixed(4)}`
          ),
        ].join("\n")
      );
      setStatus("predicted 10 steps");
    } catch (err: any) {
      setStatus(`error: ${err.message || String(err)}`);
    }
  }

  const chart = buildChartData(data, target, forecastPoints);
  const REPO_URL =
    "https://github.com/europanite/time_series_forecast-transformer";

  return (
    <View
      style={{
        flex: 1,
        backgroundColor: "#000000ff",
        padding: 24,
        alignItems: "center",
      }}
    >
      <Pressable
        onPress={() =>
          window.open(REPO_URL, "_blank", "noopener,noreferrer")
        }
      >
        <Text
          style={{
            fontSize: 24,
            fontWeight: "800",
            marginBottom: 12,
            color: "#ffffffff",
            textDecorationLine: "underline",
          }}
        >
          Time-Series Forecast Transformer
        </Text>
        <Text style={{ color: "#ffffffff", marginBottom: 16 }}>
          A browser-based multivariate time series forecasting tool. No installation, No registration, or No payment is required.
        </Text>
      </Pressable>

      {/* File input */}
      <View
        style={{
          width: "100%",
          maxWidth: 960,
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <input
          type="file"
          accept=".csv,.xlsx"
          onChange={handleFileChange}
          style={{
            padding: 8,
            borderRadius: 8,
            border: "1px solid #5a5a5a",
            background: "#111827",
            color: "#e5e7eb",
            flex: 1,
          }}
        />
        <Text style={{ marginLeft: 12, color: "#9ca3af" }}>
          Status: {status}
        </Text>
      </View>

      {/* Controls */}
      <View
        style={{
          width: "100%",
          maxWidth: 960,
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "flex-start",
          gap: 12,
          marginBottom: 16,
        }}
      >
        <Text style={{ color: "#9ca3af" }}>Target:</Text>
        <select
          value={target ?? ""}
          onChange={(e) => {
            setTarget(e.target.value || null);
            setModel(null);
            setForecast("");
            setForecastPoints([]);
          }}
          style={{
            padding: 8,
            borderRadius: 8,
            border: "1px solid #4b5563",
            background: "#111827",
            color: "#e5e7eb",
            minWidth: 120,
          }}
        >
          <option value="">(select)</option>
          {data?.headers
            .filter((h) => h !== data.datetimeKey && h !== "id")
            .map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
        </select>

        <Text style={{ color: "#9ca3af" }}>Model:</Text>
        <select
          value={modelKind}
          onChange={(e) => {
            setModelKind(e.target.value as ModelKind);
            setModel(null);
            setForecast("");
            setForecastPoints([]);
          }}
          style={{
            padding: 8,
            borderRadius: 8,
            border: "1px solid #4b5563",
            background: "#111827",
            color: "#e5e7eb",
            minWidth: 170,
          }}
        >
          <option value="seasonal_naive">Seasonal naive</option>
          <option value="chronos2">Chronos-2</option>
          <option value="timesfm">TimesFM 2.5</option>
        </select>

        <Pressable
          onPress={handleTrain}
          style={{
            width: 100,
            paddingVertical: 8,
            paddingHorizontal: 16,
            borderRadius: 999,
            backgroundColor: "#22c55e",
          }}
        >
          <Text style={{ color: "#020817", fontWeight: "600" }}>Train</Text>
        </Pressable>

        <Pressable
          onPress={handlePredict}
          style={{
            width: 100,
            paddingVertical: 8,
            paddingHorizontal: 16,
            borderRadius: 999,
            backgroundColor: model ? "#38bdf8" : "#371f1f",
            opacity: model ? 1 : 0.4,
          }}
        >
          <Text style={{ color: "#020817", fontWeight: "600" }}>
            Forecast +10
          </Text>
        </Pressable>
      </View>

      {/* Chart */}
      <View
        style={{
          width: "100%",
          maxWidth: 960,
          height: 360,
          backgroundColor: "#000000ff",
          borderRadius: 16,
          padding: 12,
        }}
      >
        {chart ? (
          <Line data={chart.data} options={chart.options} />
        ) : (
          <Text
            style={{
              color: "#6b7280",
              textAlign: "center",
              marginTop: 16,
            }}
          >
            Upload a CSV/XLSX file to visualize all series.
          </Text>
        )}
      </View>

      {/* Output */}
      <ScrollView
        style={{
          width: "100%",
          maxWidth: 960,
          marginTop: 12,
          backgroundColor: "#000000ff",
          borderRadius: 12,
          padding: 12,
        }}
      >
        <Text
          style={{
            color: "#e5e7eb",
            fontFamily: "monospace",
          }}
        >
          {forecast || "Prediction result will appear here."}
        </Text>
      </ScrollView>
    </View>
  );
}


function buildChartData(
  data: LoadedData | null,
  target: string | null,
  forecastPoints: ForecastPoint[]
): { data: ChartData<"line">; options: ChartOptions<"line"> } | null {
  if (!data || !data.rows.length) return null;

  const useDatetime =
    !!data.datetimeKey && data.headers.includes(data.datetimeKey);

  const observedLabels: string[] = useDatetime
    ? data.rows.map((r) => String(r[data.datetimeKey as string] ?? ""))
    : data.rows.map((_, i) => String(i));

  const labels = [
    ...observedLabels,
    ...forecastPoints.map((point) => point.label),
  ];

  const numericKeys = data.headers.filter((h) => {
    if (h === data.datetimeKey || h === "id") return false;
    const v = data.rows[0]?.[h];
    const n = Number(v);
    return Number.isFinite(n);
  });

  if (!numericKeys.length) return null;

  const palette = [
    "#22c55e",
    "#60a5fa",
    "#f59e0b",
    "#ef4444",
    "#a78bfa",
    "#14b8a6",
  ];

  const targetSeriesColor =
    target && numericKeys.includes(target)
      ? palette[numericKeys.indexOf(target) % palette.length]
      : palette[0];

  const datasets: LineDataset[] = numericKeys.map((k, i) => ({
    label: k,
    data: [
      ...data.rows.map((r) => Number(r[k])),
      ...forecastPoints.map(() => null),
    ],
    borderColor: palette[i % palette.length],
    backgroundColor: palette[i % palette.length],
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.25,
  }));

  const chartDatasets: LineDataset[] = [...datasets];

  if (target && forecastPoints.length > 0) {
    chartDatasets.push({
      label: `${target} forecast`,
      data: [
        ...data.rows.map(() => null),
        ...forecastPoints.map((point) => point.value),
      ],
      borderColor: targetSeriesColor,
      backgroundColor: targetSeriesColor,
      borderWidth: 2,
      borderDash: [6, 4],
      pointRadius: 3,
      tension: 0.25,
    });
  }

  return {
    data: {
      labels,
      datasets: chartDatasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          display: true,
          position: "top",
        },
      },
      scales: {
        x: { display: true },
        y: { display: true },
      },
    },
  };
}
