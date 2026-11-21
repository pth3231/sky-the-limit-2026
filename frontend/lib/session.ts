"use client";

import { useEffect, useState } from "react";
import type { components } from "./api-types";
import { loadReferenceData, loadRosters, recommend, type ReferenceData } from "./api";
import { DRAFT_ORDER, draftStateKey, fearlessExcluded, isSeriesDecided, useDraftStore } from "./draft";

type DataStatus = "loading" | "ready" | "error" | "practice";
type RecommendationStatus = "idle" | "loading" | "error" | "ready";
type DraftResponse = components["schemas"]["DraftResponse"];

const emptyData: ReferenceData = { teams: [], champions: [], patches: [] };
const practiceData: ReferenceData = {
  teams: [{ teamId: "practice-opponent", name: "Practice opponent" }],
  champions: Array.from({ length: 80 }, (_, i) => ({
    championId: `practice-${i + 1}`,
    name: `Practice champion ${String(i + 1).padStart(2, "0")}`,
  })),
  patches: ["Practice"],
};

/** Load reference data and keep recommendation results aligned with the current draft. */
export function useDraftSession() {
  const {
    setup,
    actions,
    previousGames,
    pickDisplayOrder,
    winRateHistory,
    configure,
    lock,
    undo,
    finish,
    reset,
    swapPicks,
    recordWinRate,
  } = useDraftStore();
  const [data, setData] = useState<ReferenceData>(emptyData);
  const [dataStatus, setDataStatus] = useState<DataStatus>("loading");
  const [attempt, setAttempt] = useState(0);
  const [rosterResult, setRosterResult] = useState<{
    key: string;
    data: components["schemas"]["RosterResponse"] | null;
  }>({ key: "", data: null });
  const [result, setResult] = useState<{
    key: string;
    status: RecommendationStatus;
    response: DraftResponse | null;
    error: string;
  }>({ key: "", status: "idle", response: null, error: "" });

  useEffect(() => {
    const controller = new AbortController();
    loadReferenceData(controller.signal)
      .then((loaded) => {
        if (controller.signal.aborted) return;
        setData(loaded);
        configure({ opponentTeamId: loaded.teams[0].teamId, patch: loaded.patches[0] });
        setDataStatus("ready");
      })
      .catch(() => {
        if (!controller.signal.aborted) setDataStatus("error");
      });
    return () => controller.abort();
  }, [attempt, configure]);

  const rosterKey = dataStatus === "ready" ? setup.opponentTeamId : "";
  useEffect(() => {
    if (!rosterKey) return;
    const controller = new AbortController();
    loadRosters(rosterKey, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setRosterResult({ key: rosterKey, data });
      })
      .catch(() => {
        if (!controller.signal.aborted) setRosterResult({ key: rosterKey, data: null });
      });
    return () => controller.abort();
  }, [rosterKey]);

  const seriesDecided = isSeriesDecided(setup.seriesFormat, previousGames);
  const requestKey =
    dataStatus === "ready" && setup.opponentTeamId && !seriesDecided
      ? JSON.stringify({ ...setup, patch: setup.patch || null, previousGames, actions })
      : "";

  useEffect(() => {
    if (!requestKey) return;
    const controller = new AbortController();
    setResult({ key: requestKey, status: "loading", response: null, error: "" });
    const draft = JSON.parse(requestKey) as components["schemas"]["DraftRequest"];
    recommend(draft, controller.signal)
      .then((response) => {
        if (!controller.signal.aborted) {
          const stateKey = draftStateKey(actions, previousGames);
          recordWinRate(stateKey, response.gameWinRate * 100);
          setResult({ key: requestKey, status: "ready", response, error: "" });
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setResult({
            key: requestKey,
            status: "error",
            response: null,
            error: error instanceof Error ? error.message : "Unable to load recommendations",
          });
        }
      });
    return () => controller.abort();
  }, [requestKey]);

  const activeResult = requestKey && result.key === requestKey ? result : null;
  return {
    setup,
    onSetupChange: configure,
    ...data,
    dataStatus,
    onPractice: () => {
      reset();
      configure({ opponentTeamId: practiceData.teams[0].teamId, patch: practiceData.patches[0] });
      setData(practiceData);
      setDataStatus("practice");
    },
    onRetry: () => {
      reset();
      setData(emptyData);
      setDataStatus("loading");
      setAttempt((value) => value + 1);
    },
    actions,
    previousGames,
    pickDisplayOrder,
    onSwapPicks: swapPicks,
    rosters: rosterKey && rosterResult.key === rosterKey ? rosterResult.data : null,
    order: DRAFT_ORDER,
    fearlessExcluded: fearlessExcluded(previousGames),
    seriesDecided,
    onLock: (id: string) =>
      lock(
        id,
        data.champions.map((champion) => champion.championId),
      ),
    onUndo: undo,
    onFinish: finish,
    recommendation: activeResult?.response ?? null,
    recommendationStatus: activeResult?.status ?? (requestKey ? "loading" : "idle"),
    recommendationError: activeResult?.error ?? "",
    winRateHistory,
  };
}
