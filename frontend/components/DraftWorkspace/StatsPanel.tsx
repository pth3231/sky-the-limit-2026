import type { components } from "../../lib/api-types";

type DataStatus = "loading" | "ready" | "error" | "practice";

export interface StatsPanelProps {
  recommendation: components["schemas"]["DraftResponse"] | null;
  recommendationStatus: "idle" | "loading" | "error" | "ready";
  recommendationError: string;
  winRateHistory: { key: string; depth: number; winRate: number }[];
  dataStatus: DataStatus;
}

/** Stats + guidance panel: win rate graph and short reasoning. */
export function StatsPanel({
  recommendation,
  recommendationStatus,
  recommendationError,
  winRateHistory,
  dataStatus,
}: StatsPanelProps) {
  const hasData = recommendationStatus === "ready" && recommendation;
  const isLoading = recommendationStatus === "loading";

  return (
    <aside className="stats-panel" aria-label="Stats and guidance">
      {/* Win rate chart */}
      <div className="stats-panel-section">
        <h3 className="stats-panel-label">Win Rate</h3>
        {winRateHistory.length > 0 ? (
          <WinRateChart history={winRateHistory} />
        ) : (
          <p className="empty-message">Win rate will appear after picks are made.</p>
        )}
      </div>

      {/* Guidance */}
      <div className="stats-panel-section">
        <h3 className="stats-panel-label">Guidance</h3>
        {isLoading && <p className="loading-message">Analyzing...</p>}
        {recommendationError && <p className="strip-error">{recommendationError}</p>}
        {!isLoading && !recommendationError && !hasData && (
          <p className="empty-message">
            {dataStatus === "practice"
              ? "Guidance available when connected to live data."
              : "Guidance will appear once the draft begins."}
          </p>
        )}
        {hasData && recommendation.recommendation && (
          <GuidanceBlock candidate={recommendation.recommendation} />
        )}
        {hasData && recommendation.alternatives && recommendation.alternatives.length > 0 && (
          <AlternativesList alternatives={recommendation.alternatives} />
        )}
      </div>

      {/* Chain recommendation */}
      {hasData && recommendation.chainRecommendation && (
        <div className="stats-panel-section">
          <h3 className="stats-panel-label">
            Draft Chain
            <span className="chain-engine">({recommendation.chainRecommendation.engine})</span>
          </h3>
          <p className="chain-estimated-win">
            Est. win rate: <strong>{(recommendation.chainRecommendation.estimatedWinRate * 100).toFixed(1)}%</strong>
          </p>
          <ChainSteps steps={recommendation.chainRecommendation.steps} />
        </div>
      )}
    </aside>
  );
}

function WinRateChart({ history }: { history: { key: string; depth: number; winRate: number }[] }) {
  if (history.length === 0) return null;
  const width = 400;
  const height = 200;
  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const values = history.map((h) => h.winRate);

  // Y-axis: fixed 0-100% range
  const yMin = 0;
  const yMax = 100;
  const yRange = yMax - yMin;

  // X-axis: pick numbers
  const xMax = Math.max(values.length - 1, 1);

  const points = values.map((v, i) => {
    const x = padding.left + (i / xMax) * chartWidth;
    const y = padding.top + (1 - (v - yMin) / yRange) * chartHeight;
    return { x, y, value: v };
  });

  const polylinePoints = points.map((p) => `${p.x},${p.y}`).join(" ");

  // Grid lines at 25%, 50%, 75%, 100%
  const gridLines = [25, 50, 75, 100].map((val) => {
    const y = padding.top + (1 - (val - yMin) / yRange) * chartHeight;
    return { y, label: `${val}%` };
  });

  return (
    <div className="win-rate-chart-container">
      <svg
        className="win-rate-chart"
        viewBox={`0 0 ${width} ${height}`}
        aria-label="Win rate trend"
        role="img"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid lines */}
        {gridLines.map((line, i) => (
          <g key={i}>
            <line
              x1={padding.left}
              y1={line.y}
              x2={width - padding.right}
              y2={line.y}
              stroke="var(--line)"
              strokeWidth="1"
              strokeDasharray="4,4"
            />
            <text
              x={padding.left - 8}
              y={line.y + 4}
              textAnchor="end"
              fill="var(--muted)"
              fontSize="11"
              fontFamily="Consolas, monospace"
            >
              {line.label}
            </text>
          </g>
        ))}

        {/* X-axis labels */}
        {points.map((p, i) => (
          <text
            key={i}
            x={p.x}
            y={height - padding.bottom + 20}
            textAnchor="middle"
            fill="var(--muted)"
            fontSize="11"
            fontFamily="Consolas, monospace"
          >
            {i + 1}
          </text>
        ))}

        {/* X-axis label */}
        <text
          x={padding.left + chartWidth / 2}
          y={height - 4}
          textAnchor="middle"
          fill="var(--muted)"
          fontSize="11"
          fontFamily="Arial, sans-serif"
        >
          Pick number
        </text>

        {/* Area fill under the line */}
        <polygon
          points={`${points[0].x},${padding.top + chartHeight} ${polylinePoints} ${points[points.length - 1].x},${padding.top + chartHeight}`}
          fill="var(--cloud9)"
          opacity="0.1"
        />

        {/* Line */}
        <polyline
          points={polylinePoints}
          fill="none"
          stroke="var(--cloud9)"
          strokeWidth="2.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Data points with tooltips */}
        {points.map((p, i) => (
          <g key={i}>
            <circle
              cx={p.x}
              cy={p.y}
              r="5"
              fill="var(--ground)"
              stroke="var(--cloud9)"
              strokeWidth="2"
            />
            <title>{`Pick ${i + 1}: ${p.value.toFixed(1)}% win rate`}</title>
          </g>
        ))}

        {/* Current value label on last point */}
        {points.length > 0 && (
          <g>
            <rect
              x={points[points.length - 1].x - 30}
              y={points[points.length - 1].y - 28}
              width="60"
              height="20"
              fill="var(--surface)"
              stroke="var(--cloud9)"
              strokeWidth="1"
            />
            <text
              x={points[points.length - 1].x}
              y={points[points.length - 1].y - 14}
              textAnchor="middle"
              fill="var(--cloud9)"
              fontSize="12"
              fontWeight="700"
              fontFamily="Consolas, monospace"
            >
              {points[points.length - 1].value.toFixed(1)}%
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}

function GuidanceBlock({ candidate }: { candidate: components["schemas"]["Candidate"] }) {
  return (
    <div>
      <p>
        <strong>{candidate.championName}</strong>
        <span className="strip-score"> · score {candidate.score.toFixed(2)}</span>
      </p>
      {candidate.reason && <p className="strip-reason">{candidate.reason}</p>}
      {candidate.contributingFactors && candidate.contributingFactors.length > 0 && (
        <ul className="factor-list">
          {candidate.contributingFactors.map((factor, i) => (
            <li key={i}>
              <span>{factor.label}</span>
              <b>{factor.impact.toFixed(2)}</b>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function AlternativesList({ alternatives }: { alternatives: components["schemas"]["Candidate"][] }) {
  return (
    <ul className="alternative-list">
      {alternatives.slice(0, 5).map((alt) => (
        <li key={alt.championId}>
          <div>
            <span>{alt.championName}</span>
            <b>{alt.score.toFixed(2)}</b>
          </div>
          {alt.reason && <p>{alt.reason}</p>}
        </li>
      ))}
    </ul>
  );
}

function ChainSteps({ steps }: { steps: components["schemas"]["ChainStep"][] }) {
  if (steps.length === 0) return null;
  return (
    <ol className="chain-steps-list">
      {steps.map((step) => (
        <li
          key={step.slotIndex}
          className={`chain-step chain-step-${step.action}`}
        >
          <span className="chain-step-action">
            {step.action === "ban" ? "🚫" : "✓"} {step.action.toUpperCase()} #{step.slotIndex + 1}
          </span>
          <span className="chain-step-champion">{step.championName}</span>
          <span className="chain-step-score">{step.score.toFixed(2)}</span>
        </li>
      ))}
    </ol>
  );
}
