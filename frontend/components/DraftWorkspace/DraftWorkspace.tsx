"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { components } from "../../lib/api-types";
import { ChampionSelector } from "./ChampionSelector";
import { DraftBar } from "./DraftBar";
import { StatsPanel } from "./StatsPanel";
import type { Setup, Side } from "./TopBar";

type Result = "win" | "loss";

/** Controlled draft state and callbacks supplied by the workspace owner. */
export interface DraftWorkspaceProps {
  setup: Setup;
  onSetupChange(patch: Partial<Setup>): void;
  teams: { teamId: string; name: string }[];
  champions: {
    championId: string;
    name: string;
    role?: components["schemas"]["ChampionSummary"]["role"];
    portraitUrl?: components["schemas"]["ChampionSummary"]["portraitUrl"];
  }[];
  patches: string[];
  dataStatus: "loading" | "ready" | "error" | "practice";
  onPractice(): void;
  onRetry(): void;
  actions: string[];
  previousGames: { ourPicks: string[]; opponentPicks: string[]; result: Result }[];
  order: { index: number; side: Side; action: "ban" | "pick" }[];
  fearlessExcluded: string[];
  seriesDecided: boolean;
  onLock(id: string): void;
  onUndo(): void;
  onFinish(result: Result): void;
  recommendation: components["schemas"]["DraftResponse"] | null;
  recommendationStatus: "idle" | "loading" | "error" | "ready";
  recommendationError: string;
  rosters?: { ourTeam: components["schemas"]["TeamRoster"]; opponentTeam: components["schemas"]["TeamRoster"] } | null;
  pickDisplayOrder?: number[];
  winRateHistory?: { key: string; depth: number; winRate: number }[];
  onSwapPicks?(first: number, second: number): void;
}

/** Render the draft workspace matching redesign_overall.png: selector | stats / draft bar. */
export function DraftWorkspace({
  setup,
  onSetupChange,
  teams,
  champions,
  patches,
  dataStatus,
  onPractice,
  onRetry,
  actions,
  previousGames,
  order,
  fearlessExcluded,
  seriesDecided,
  onLock,
  onUndo,
  onFinish,
  recommendation,
  recommendationStatus,
  recommendationError,
  rosters = null,
  pickDisplayOrder = order.map((slot) => slot.index),
  winRateHistory = [],
  onSwapPicks = () => {},
}: DraftWorkspaceProps) {
  const opponentName =
    teams.find((team) => team.teamId === setup.opponentTeamId)?.name ?? "Opponent";
  const currentSlot = order.find((slot) => slot.index === actions.length);
  const complete = actions.length === 20;
  const canDraft = (dataStatus === "ready" || dataStatus === "practice") && !seriesDecided;
  const activeTeam = currentSlot?.side === setup.ourSide ? "Cloud9" : opponentName;
  const slotActions = actions.map(
    (_, index) => order.find((slot) => slot.index === index)?.action ?? "pick",
  );
  const response =
    dataStatus === "ready" && recommendationStatus === "ready" ? recommendation : null;
  const candidates = response
    ? [
        ...(response.recommendation ? [response.recommendation] : []),
        ...response.alternatives,
      ].filter((c, i, all) => all.findIndex((o) => o.championId === c.championId) === i)
    : [];

  // Determine the current action label for the header
  const actionLabel = seriesDecided
    ? "SERIES DECIDED"
    : complete
      ? "DRAFT COMPLETE"
      : currentSlot
        ? `${currentSlot.action.toUpperCase()} A CHAMPION!`
        : "WAITING";

  // Resizable panel state
  const [panelWidth, setPanelWidth] = useState(320); // stats panel width
  const [draftBarHeight, setDraftBarHeight] = useState(160); // draft bar height
  const isDraggingVertical = useRef(false);
  const isDraggingHorizontal = useRef(false);
  const lastMouseX = useRef(0);
  const lastMouseY = useRef(0);

  const handleVerticalMouseDown = useCallback((e: React.MouseEvent) => {
    isDraggingVertical.current = true;
    lastMouseX.current = e.clientX;
    e.preventDefault();
  }, []);

  const handleHorizontalMouseDown = useCallback((e: React.MouseEvent) => {
    isDraggingHorizontal.current = true;
    lastMouseY.current = e.clientY;
    e.preventDefault();
  }, []);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isDraggingVertical.current) {
      const delta = lastMouseX.current - e.clientX;
      lastMouseX.current = e.clientX;
      setPanelWidth((prev) => Math.max(240, Math.min(600, prev + delta)));
    }
    if (isDraggingHorizontal.current) {
      const delta = lastMouseY.current - e.clientY;
      lastMouseY.current = e.clientY;
      setDraftBarHeight((prev) => Math.max(100, Math.min(300, prev + delta)));
    }
  }, []);

  const handleMouseUp = useCallback(() => {
    isDraggingVertical.current = false;
    isDraggingHorizontal.current = false;
  }, []);

  // Attach global mouse move/up listeners
  const workspaceRef = useRef<HTMLElement>(null);
  useEffect(() => {
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  return (
    <main ref={workspaceRef} className="workspace">
      {/* Header with title */}
      <header className="draft-header">
        <h1 className="action-label">Draft Recommendation</h1>
        {!seriesDecided && !complete && currentSlot && (
          <p className="active-team">
            {actionLabel} · {activeTeam}
          </p>
        )}
        {seriesDecided && <p className="active-team">Series decided</p>}
        {complete && !seriesDecided && <p className="active-team">Draft complete</p>}
      </header>

      {/* Main draft area - selector | resize handle | stats */}
      <div className="draft-screen">
        {/* Left: Champion selector */}
        <section
          className="pool-section"
          aria-label="Champion pool"
          style={{ flexBasis: `calc(100% - ${panelWidth}px - 6px)`, maxWidth: `calc(100% - ${panelWidth}px - 6px)` }}
        >
          <ChampionSelector
            champions={champions}
            actions={actions}
            slotActions={slotActions}
            fearlessExcluded={fearlessExcluded}
            candidates={candidates}
            currentAction={currentSlot?.action}
            locked={!canDraft || complete || !currentSlot}
            onLock={onLock}
          />
        </section>

        {/* Vertical resize handle */}
        <div
          className="resize-handle resize-handle-vertical"
          onMouseDown={handleVerticalMouseDown}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize champion selector and stats panel"
          tabIndex={0}
        />

        {/* Right: Stats + guidance panel */}
        <div className="stats-panel-wrapper" style={{ width: `${panelWidth}px` }}>
          <StatsPanel
            recommendation={response}
            recommendationStatus={recommendationStatus}
            recommendationError={recommendationError}
            winRateHistory={winRateHistory}
            dataStatus={dataStatus}
          />
        </div>
      </div>

      {/* Horizontal resize handle */}
      <div
        className="resize-handle resize-handle-horizontal"
        onMouseDown={handleHorizontalMouseDown}
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize main area and draft bar"
        tabIndex={0}
      />

      {/* Bottom: Draft bar with both teams */}
      <div className="draft-bar-wrapper" style={{ height: `${draftBarHeight}px` }}>
        <DraftBar
          ourTeam="Cloud9"
          opponentTeam={opponentName}
          order={order}
          actions={actions}
          champions={champions}
          rosters={rosters}
          canDraft={canDraft}
          seriesDecided={seriesDecided}
          complete={complete}
          onUndo={onUndo}
          onFinish={onFinish}
          onSwapPicks={onSwapPicks}
          displayOrder={pickDisplayOrder}
        />
      </div>
    </main>
  );
}
