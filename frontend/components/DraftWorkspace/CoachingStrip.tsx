"use client";

import type { components } from "../../lib/api-types";
import type { WinRatePoint } from "../../lib/draft";

type DraftResponse = components["schemas"]["DraftResponse"];

const percentage = (value: number) => `${Number((value * 100).toFixed(1))}%`;

/** Small SVG line chart showing win-rate history across draft actions. */
export function WinRateChart({ points }: { points: WinRatePoint[] }) {
  if (points.length < 2) return null;

  const sorted = [...points].sort((a, b) => a.depth - b.depth);
  const width = 200;
  const height = 40;
  const padding = 4;

  const minRate = Math.min(...sorted.map((p) => p.winRate), 40);
  const maxRate = Math.max(...sorted.map((p) => p.winRate), 60);
  const range = maxRate - minRate || 1;

  const coords = sorted.map((p, i) => ({
    x: padding + (i / (sorted.length - 1 || 1)) * (width - padding * 2),
    y: height - padding - ((p.winRate - minRate) / range) * (height - padding * 2),
  }));

  const pathD = coords.map((c, i) => `${i === 0 ? "M" : "L"}${c.x},${c.y}`).join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="win-rate-chart"
      role="img"
      aria-label={`Win rate trend: ${sorted.map((p) => percentage(p.winRate)).join(", ")}`}
    >
      <path d={pathD} fill="none" stroke="currentColor" strokeWidth="2" />
      {coords.map((c, i) => (
        <circle key={i} cx={c.x} cy={c.y} r="3" fill="currentColor" />
      ))}
    </svg>
  );
}

/** Compact coaching strip: win estimate, trend chart, and short recommendation. */
export function CoachingStrip({
  response,
  winRateHistory,
  dataStatus,
  recommendationStatus,
  recommendationError,
}: {
  response: DraftResponse | null;
  winRateHistory: WinRatePoint[];
  dataStatus: "loading" | "ready" | "error" | "practice";
  recommendationStatus: "idle" | "loading" | "error" | "ready";
  recommendationError: string;
}) {
  const statusLabel =
    dataStatus === "practice"
      ? "PRACTICE"
      : dataStatus === "error" || recommendationStatus === "error"
        ? "OFFLINE"
        : dataStatus === "loading" || recommendationStatus === "loading"
          ? "LOADING"
          : "LIVE";

  if (!response) {
    return (
      <div className="coaching-strip" aria-live="polite">
        <span className={`strip-tag ${statusLabel.toLowerCase()}`}>{statusLabel}</span>
        {dataStatus === "practice" ? (
          <p className="strip-empty">Insights are unavailable in practice mode.</p>
        ) : dataStatus !== "ready" ? (
          <p className="strip-empty">Insights will appear when draft data is available.</p>
        ) : recommendationStatus === "loading" ? (
          <p className="strip-empty">Loading recommendations…</p>
        ) : recommendationStatus === "error" ? (
          <div role="alert">
            <p className="strip-empty">Could not load recommendations.</p>
            <p className="strip-error">{recommendationError || "Please try again."}</p>
          </div>
        ) : (
          <p className="strip-empty">Recommendations appear when the draft is ready.</p>
        )}
      </div>
    );
  }

  const engineLabel =
    response.engine === "model"
      ? "MODEL"
      : response.engine === "statistical"
        ? "STATISTICAL"
        : "ESTIMATE";

  return (
    <div className="coaching-strip">
      <div className="strip-estimate">
        <span className={`strip-tag ${statusLabel.toLowerCase()}`}>{statusLabel}</span>
        <div className="estimate-value">
          <span className="micro-label">{engineLabel} WIN CHANCE</span>
          <strong>{percentage(response.gameWinRate)}</strong>
        </div>
        <WinRateChart points={winRateHistory} />
      </div>
      <div className="strip-recommendation">
        {response.recommendation ? (
          <>
            <span className="micro-label">RECOMMENDED</span>
            <strong>{response.recommendation.championName}</strong>
            <span className="strip-score">{percentage(response.recommendation.score)}</span>
            <p className="strip-reason">{response.recommendation.reason}</p>
          </>
        ) : (
          <p className="strip-empty">No recommendation for this slot.</p>
        )}
      </div>
    </div>
  );
}
