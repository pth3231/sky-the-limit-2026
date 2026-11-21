"use client";

export type Side = "blue" | "red";
export type SeriesFormat = "bo1" | "bo3" | "bo5";
export type Setup = {
  opponentTeamId: string;
  seriesFormat: SeriesFormat;
  patch: string;
  ourSide: Side;
};

export interface TopBarProps {
  setup: Setup;
  onSetupChange(patch: Partial<Setup>): void;
  teams: { teamId: string; name: string }[];
  patches: string[];
  dataStatus: "loading" | "ready" | "error" | "practice";
  onPractice(): void;
  onRetry(): void;
  opponentName: string;
  wins: number;
  losses: number;
  gameNumber: number;
  lockedCount: number;
  seriesDecided: boolean;
}

/** Slim top bar: brand, compact series setup, series score and data status. */
export function TopBar({
  setup,
  onSetupChange,
  teams,
  patches,
  dataStatus,
  onPractice,
  onRetry,
  opponentName,
  wins,
  losses,
  gameNumber,
  lockedCount,
  seriesDecided,
}: TopBarProps) {
  return (
    <header className="topbar">
      <div className="topbar-row">
        <div className="brand">
          <span className="brand-mark">C9</span>
          <span>
            Draft <b>Coach</b>
          </span>
        </div>
        <div className="setup-controls" role="group" aria-label="Series setup">
          <label>
            Opponent
            <select
              value={setup.opponentTeamId}
              disabled={dataStatus === "loading" || dataStatus === "error"}
              onChange={(event) => onSetupChange({ opponentTeamId: event.target.value })}
            >
              <option value="">Select opponent</option>
              {teams.map((team) => (
                <option key={team.teamId} value={team.teamId}>
                  {team.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Series format
            <select
              value={setup.seriesFormat}
              onChange={(event) =>
                onSetupChange({ seriesFormat: event.target.value as SeriesFormat })
              }
            >
              <option value="bo1">Best of 1</option>
              <option value="bo3">Best of 3</option>
              <option value="bo5">Best of 5</option>
            </select>
          </label>
          <label>
            Patch
            <select
              value={setup.patch}
              disabled={dataStatus === "loading" || dataStatus === "error"}
              onChange={(event) => onSetupChange({ patch: event.target.value })}
            >
              <option value="">Select patch</option>
              {patches.map((patch) => (
                <option key={patch} value={patch}>
                  {patch}
                </option>
              ))}
            </select>
          </label>
          <label>
            Cloud9 side
            <select
              value={setup.ourSide}
              onChange={(event) => onSetupChange({ ourSide: event.target.value as Side })}
            >
              <option value="blue">Blue side</option>
              <option value="red">Red side</option>
            </select>
          </label>
        </div>
        <div
          className="series-score"
          aria-label={`Series score: Cloud9 ${wins}, ${opponentName} ${losses}`}
        >
          <span className="micro-label">SCORE</span>
          <span className="accent-text">{wins}</span>
          <span className="score-separator">:</span>
          <span className="opponent-text">{losses}</span>
        </div>
        <div className="status-chip" role="status">
          {seriesDecided ? (
            <span>SERIES COMPLETE</span>
          ) : (
            <>
              <span>GAME {gameNumber}</span>
              <span>{lockedCount} OF 20 LOCKED</span>
            </>
          )}
        </div>
      </div>
      {dataStatus === "loading" && (
        <p className="status-banner" role="status">
          Loading draft data…
        </p>
      )}
      {dataStatus === "error" && (
        <div className="status-banner status-error" role="alert">
          <span>Draft data is unavailable. Retry or continue in practice mode.</span>
          <div className="status-actions">
            <button className="button button-outline" onClick={onRetry}>
              Retry
            </button>
            <button className="button button-primary" onClick={onPractice}>
              Practice mode
            </button>
          </div>
        </div>
      )}
      {dataStatus === "practice" && (
        <p className="status-banner" role="status">
          Practice mode: no recommendations or statistical estimates are available.
        </p>
      )}
    </header>
  );
}
