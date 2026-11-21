import { create } from "zustand";
import type { components } from "./api-types";

type PreviousGame = components["schemas"]["PreviousGame"];
type Side = "blue" | "red";
type SeriesFormat = "bo1" | "bo3" | "bo5";

export interface WinRatePoint {
  key: string;
  depth: number;
  winRate: number;
}

/** Build a deterministic key from the draft state for stale-response detection. */
export function draftStateKey(actions: string[], previousGames: PreviousGame[]): string {
  const gameSegment = previousGames
    .map((g) => `${g.ourPicks.join(",")}|${g.opponentPicks.join(",")}`)
    .join(";");
  return `${gameSegment}||${actions.join(",")}`;
}

/** Prune history to keep only points that are still on the current action path. */
function pruneHistory(
  history: WinRatePoint[],
  actions: string[],
  previousGames: PreviousGame[],
): WinRatePoint[] {
  const currentKey = draftStateKey(actions, previousGames);
  const actionPart = currentKey.split("||")[1] ?? "";
  const currentActions = actionPart.split(",").filter(Boolean);
  return history.filter((point) => {
    if (point.depth > currentActions.length) return false;
    const pointActionPart = point.key.split("||")[1] ?? "";
    const pointActions = pointActionPart.split(",").filter(Boolean);
    if (pointActions.length > currentActions.length) return false;
    for (let i = 0; i < pointActions.length; i++) {
      if (pointActions[i] !== currentActions[i]) return false;
    }
    return true;
  });
}

/** Coach-selected series settings. */
export interface Setup {
  opponentTeamId: string;
  seriesFormat: SeriesFormat;
  patch: string;
  ourSide: Side;
}

type DraftSlot = { index: number; side: Side; action: "pick" | "ban" };

/** Standard professional pick/ban order, matching the backend. */
export const DRAFT_ORDER: DraftSlot[] = (
  [
    ["ban", "BRBRBR"],
    ["pick", "BRRBBR"],
    ["ban", "RBRB"],
    ["pick", "RBBR"],
  ] as const
)
  .flatMap(([action, sides]) =>
    [...sides].map((side) => ({
      action,
      side: side === "B" ? ("blue" as const) : ("red" as const),
    })),
  )
  .map((slot, index) => ({ ...slot, index }));

/** Return the picks from completed games, which cannot be drafted again. */
export function fearlessExcluded(games: PreviousGame[]): string[] {
  return games.flatMap((game) => [...game.ourPicks, ...game.opponentPicks]);
}

/** Determine whether either team has won the series. */
export function isSeriesDecided(format: SeriesFormat, games: PreviousGame[]): boolean {
  const needed = { bo1: 1, bo3: 2, bo5: 3 }[format];
  return (
    games.filter((game) => game.result === "win").length >= needed ||
    games.filter((game) => game.result === "loss").length >= needed
  );
}

const defaultSetup: Setup = { opponentTeamId: "", seriesFormat: "bo3", patch: "", ourSide: "blue" };
const defaultPickDisplayOrder = () => DRAFT_ORDER.map((slot) => slot.index);

interface DraftState {
  setup: Setup;
  actions: string[];
  previousGames: PreviousGame[];
  pickDisplayOrder: number[];
  winRateHistory: WinRatePoint[];
  configure: (changes: Partial<Setup>) => void;
  lock: (champion: string, known: string[]) => void;
  undo: () => void;
  finish: (result: "win" | "loss") => void;
  swapPicks: (first: number, second: number) => void;
  recordWinRate: (requestKey: string, winRate: number) => void;
  reset: () => void;
}

/** Client-side draft state; the server receives the complete state on every request. */
export const useDraftStore = create<DraftState>((set) => ({
  setup: { ...defaultSetup },
  actions: [],
  previousGames: [],
  pickDisplayOrder: defaultPickDisplayOrder(),
  winRateHistory: [],
  configure: (changes) =>
    set((state) => {
      const setup = { ...state.setup, ...changes };
      if (
        Object.keys(changes).every(
          (key) => setup[key as keyof Setup] === state.setup[key as keyof Setup],
        )
      )
        return state;
      return {
        setup,
        actions: [],
        previousGames: [],
        pickDisplayOrder: defaultPickDisplayOrder(),
        winRateHistory: [],
      };
    }),
  lock: (champion, known) =>
    set((state) => {
      if (
        !known.includes(champion) ||
        state.actions.includes(champion) ||
        fearlessExcluded(state.previousGames).includes(champion) ||
        state.actions.length >= DRAFT_ORDER.length ||
        isSeriesDecided(state.setup.seriesFormat, state.previousGames)
      )
        return state;
      const newActions = [...state.actions, champion];
      return {
        actions: newActions,
        winRateHistory: pruneHistory(state.winRateHistory, newActions, state.previousGames),
      };
    }),
  undo: () =>
    set((state) => {
      const newActions = state.actions.slice(0, -1);
      return {
        actions: newActions,
        pickDisplayOrder: defaultPickDisplayOrder(),
        winRateHistory: pruneHistory(state.winRateHistory, newActions, state.previousGames),
      };
    }),
  swapPicks: (first, second) =>
    set((state) => {
      const a = DRAFT_ORDER[first];
      const b = DRAFT_ORDER[second];
      if (
        state.actions.length !== DRAFT_ORDER.length ||
        !a ||
        !b ||
        a.action !== "pick" ||
        b.action !== "pick" ||
        a.side !== b.side ||
        first === second
      )
        return state;
      const pickDisplayOrder = [...state.pickDisplayOrder];
      [pickDisplayOrder[first], pickDisplayOrder[second]] = [
        pickDisplayOrder[second],
        pickDisplayOrder[first],
      ];
      return { pickDisplayOrder };
    }),
  recordWinRate: (requestKey, winRate) =>
    set((state) => {
      if (!Number.isFinite(winRate)) return state;
      const currentKey = draftStateKey(state.actions, state.previousGames);
      if (requestKey !== currentKey) return state;
      const depth = state.actions.length;
      const existing = state.winRateHistory.filter((p) => p.depth !== depth);
      return { winRateHistory: [...existing, { key: currentKey, depth, winRate }] };
    }),
  finish: (result) =>
    set((state) => {
      if (
        state.actions.length !== DRAFT_ORDER.length ||
        isSeriesDecided(state.setup.seriesFormat, state.previousGames)
      )
        return state;
      const ourPicks: string[] = [];
      const opponentPicks: string[] = [];
      DRAFT_ORDER.forEach((slot, index) => {
        if (slot.action === "pick") {
          (slot.side === state.setup.ourSide ? ourPicks : opponentPicks).push(state.actions[index]);
        }
      });
      return {
        previousGames: [...state.previousGames, { ourPicks, opponentPicks, result }],
        actions: [],
        pickDisplayOrder: defaultPickDisplayOrder(),
        winRateHistory: [],
      };
    }),
  reset: () =>
    set({
      setup: { ...defaultSetup },
      actions: [],
      previousGames: [],
      pickDisplayOrder: defaultPickDisplayOrder(),
      winRateHistory: [],
    }),
}));
