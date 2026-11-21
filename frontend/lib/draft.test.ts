import { beforeEach, describe, expect, it } from "vitest";
import {
  DRAFT_ORDER,
  draftStateKey,
  fearlessExcluded,
  isSeriesDecided,
  useDraftStore,
} from "./draft";

const champions = Array.from({ length: 60 }, (_, index) => `champion-${index}`);

function completeDraft(offset = 0) {
  for (const champion of champions.slice(offset, offset + 20)) {
    useDraftStore.getState().lock(champion, champions);
  }
}

describe("draft state", () => {
  beforeEach(() => useDraftStore.getState().reset());

  it("matches the professional 20-slot order", () => {
    expect(DRAFT_ORDER.map((slot) => `${slot.action}:${slot.side}`)).toEqual([
      "ban:blue",
      "ban:red",
      "ban:blue",
      "ban:red",
      "ban:blue",
      "ban:red",
      "pick:blue",
      "pick:red",
      "pick:red",
      "pick:blue",
      "pick:blue",
      "pick:red",
      "ban:red",
      "ban:blue",
      "ban:red",
      "ban:blue",
      "pick:red",
      "pick:blue",
      "pick:blue",
      "pick:red",
    ]);
  });

  it("rejects unknown and repeated champions and allows undo", () => {
    const state = useDraftStore.getState();
    state.lock("unknown", champions);
    state.lock(champions[0], champions);
    state.lock(champions[0], champions);
    expect(useDraftStore.getState().actions).toEqual([champions[0]]);
    state.undo();
    expect(useDraftStore.getState().actions).toEqual([]);
  });

  it("records complete games only and carries picks, not bans, into Fearless", () => {
    useDraftStore.getState().finish("win");
    expect(useDraftStore.getState().previousGames).toHaveLength(0);
    completeDraft();
    useDraftStore.getState().finish("win");
    const state = useDraftStore.getState();
    expect(state.previousGames[0].ourPicks).toHaveLength(5);
    expect(state.previousGames[0].opponentPicks).toHaveLength(5);
    expect(fearlessExcluded(state.previousGames)).toHaveLength(10);
    state.lock(champions[6], champions);
    state.lock(champions[0], champions);
    expect(useDraftStore.getState().actions).toEqual([champions[0]]);
  });

  it("honors the configured side when saving picks", () => {
    useDraftStore.getState().configure({ ourSide: "red" });
    completeDraft();
    useDraftStore.getState().finish("loss");
    expect(useDraftStore.getState().previousGames[0].ourPicks).toEqual([
      champions[7],
      champions[8],
      champions[11],
      champions[16],
      champions[19],
    ]);
  });

  it("stops a decided series and prevents further moves", () => {
    useDraftStore.getState().configure({ seriesFormat: "bo1" });
    completeDraft();
    useDraftStore.getState().finish("win");
    const state = useDraftStore.getState();
    expect(isSeriesDecided(state.setup.seriesFormat, state.previousGames)).toBe(true);
    state.lock(champions[20], champions);
    expect(useDraftStore.getState().actions).toHaveLength(0);
  });

  it("resets games and actions when setup changes but not for identical values", () => {
    completeDraft();
    useDraftStore.getState().finish("win");
    useDraftStore.getState().lock(champions[20], champions);
    useDraftStore.getState().configure({ seriesFormat: "bo3" });
    expect(useDraftStore.getState().previousGames).toHaveLength(1);
    useDraftStore.getState().configure({ patch: "different" });
    expect(useDraftStore.getState().previousGames).toHaveLength(0);
    expect(useDraftStore.getState().actions).toHaveLength(0);
  });

  it("swaps only completed picks on the same side without changing draft or recorded results", () => {
    const identity = DRAFT_ORDER.map((slot) => slot.index);
    useDraftStore.getState().swapPicks(6, 9);
    expect(useDraftStore.getState().pickDisplayOrder).toEqual(identity);
    completeDraft();
    const original = [...useDraftStore.getState().actions];
    useDraftStore.getState().swapPicks(6, 7);
    useDraftStore.getState().swapPicks(0, 2);
    expect(useDraftStore.getState().pickDisplayOrder).toEqual(identity);
    useDraftStore.getState().swapPicks(6, 9);
    const state = useDraftStore.getState();
    expect(state.pickDisplayOrder[6]).toBe(9);
    expect(state.pickDisplayOrder[9]).toBe(6);
    expect(state.actions).toEqual(original);
    state.finish("win");
    expect(useDraftStore.getState().previousGames[0].ourPicks).toEqual(
      [6, 9, 10, 17, 18].map((i) => champions[i]),
    );
    expect(useDraftStore.getState().pickDisplayOrder).toEqual(identity);
  });

  it("resets display swaps on undo, setup changes and reset", () => {
    const identity = DRAFT_ORDER.map((slot) => slot.index);
    completeDraft();
    useDraftStore.getState().swapPicks(6, 9);
    useDraftStore.getState().undo();
    expect(useDraftStore.getState().pickDisplayOrder).toEqual(identity);
    useDraftStore.getState().lock(champions[19], champions);
    useDraftStore.getState().swapPicks(6, 9);
    useDraftStore.getState().configure({ patch: "2" });
    expect(useDraftStore.getState().pickDisplayOrder).toEqual(identity);
    completeDraft();
    useDraftStore.getState().swapPicks(6, 9);
    useDraftStore.getState().reset();
    expect(useDraftStore.getState().pickDisplayOrder).toEqual(identity);
  });
});

describe("win-rate history", () => {
  beforeEach(() => useDraftStore.getState().reset());

  function record(key: string, winRate: number) {
    useDraftStore.getState().recordWinRate(key, winRate);
  }

  function keyFor(actions: string[], previousGames: { ourPicks: string[]; opponentPicks: string[]; result: "win" | "loss" }[]) {
    return draftStateKey(actions, previousGames);
  }

  it("records a win-rate point at the current state key", () => {
    const state = useDraftStore.getState();
    const k = keyFor(state.actions, state.previousGames);
    record(k, 52.3);
    expect(useDraftStore.getState().winRateHistory).toEqual([
      { key: k, depth: 0, winRate: 52.3 },
    ]);
  });

  it("ignores non-finite values", () => {
    const state = useDraftStore.getState();
    const k = keyFor(state.actions, state.previousGames);
    record(k, NaN);
    record(k, Infinity);
    record(k, -Infinity);
    expect(useDraftStore.getState().winRateHistory).toHaveLength(0);
  });

  it("rejects stale response keys", () => {
    const state = useDraftStore.getState();
    const k1 = keyFor(state.actions, state.previousGames);
    state.lock(champions[0], champions);
    record(k1, 60);
    expect(useDraftStore.getState().winRateHistory).toHaveLength(0);
  });

  it("replaces a same-depth point on refetch", () => {
    const state = useDraftStore.getState();
    const k = keyFor(state.actions, state.previousGames);
    record(k, 50);
    record(k, 55);
    const history = useDraftStore.getState().winRateHistory;
    expect(history).toHaveLength(1);
    expect(history[0].winRate).toBe(55);
  });

  it("accumulates points across locks", () => {
    const s0 = useDraftStore.getState();
    const k0 = keyFor(s0.actions, s0.previousGames);
    record(k0, 50);
    s0.lock(champions[0], champions);
    const s1 = useDraftStore.getState();
    const k1 = keyFor(s1.actions, s1.previousGames);
    record(k1, 52);
    s1.lock(champions[1], champions);
    const s2 = useDraftStore.getState();
    const k2 = keyFor(s2.actions, s2.previousGames);
    record(k2, 54);
    const history = useDraftStore.getState().winRateHistory;
    expect(history).toHaveLength(3);
    expect(history.map((p) => p.depth)).toEqual([0, 1, 2]);
  });

  it("truncates later points on undo", () => {
    const s0 = useDraftStore.getState();
    const k0 = keyFor(s0.actions, s0.previousGames);
    record(k0, 50);
    s0.lock(champions[0], champions);
    const s1 = useDraftStore.getState();
    const k1 = keyFor(s1.actions, s1.previousGames);
    record(k1, 52);
    s1.lock(champions[1], champions);
    const s2 = useDraftStore.getState();
    const k2 = keyFor(s2.actions, s2.previousGames);
    record(k2, 54);
    s2.undo();
    const history = useDraftStore.getState().winRateHistory;
    expect(history).toHaveLength(2);
    expect(history.map((p) => p.depth)).toEqual([0, 1]);
  });

  it("replaces the surviving point after undo and record", () => {
    const s0 = useDraftStore.getState();
    const k0 = keyFor(s0.actions, s0.previousGames);
    record(k0, 50);
    s0.lock(champions[0], champions);
    const s1 = useDraftStore.getState();
    const k1 = keyFor(s1.actions, s1.previousGames);
    record(k1, 52);
    s1.undo();
    // After undo, the depth-1 point is pruned (action prefix mismatch).
    // The current key matches depth 0, so recording replaces that point.
    const s2 = useDraftStore.getState();
    const kAfterUndo = keyFor(s2.actions, s2.previousGames);
    record(kAfterUndo, 53);
    const history = useDraftStore.getState().winRateHistory;
    expect(history).toHaveLength(1);
    expect(history[0].depth).toBe(0);
    expect(history[0].winRate).toBe(53);
  });

  it("clears later points on branching", () => {
    const s0 = useDraftStore.getState();
    const k0 = keyFor(s0.actions, s0.previousGames);
    record(k0, 50);
    s0.lock(champions[0], champions);
    const s1 = useDraftStore.getState();
    const k1 = keyFor(s1.actions, s1.previousGames);
    record(k1, 52);
    s1.lock(champions[1], champions);
    const s2 = useDraftStore.getState();
    const k2 = keyFor(s2.actions, s2.previousGames);
    record(k2, 54);
    s2.undo();
    useDraftStore.getState().undo();
    useDraftStore.getState().lock(champions[2], champions);
    const s3 = useDraftStore.getState();
    const k1b = keyFor(s3.actions, s3.previousGames);
    record(k1b, 51);
    const history = useDraftStore.getState().winRateHistory;
    expect(history).toHaveLength(2);
    expect(history.map((p) => p.depth)).toEqual([0, 1]);
    expect(history[1].winRate).toBe(51);
  });

  it("clears history on configure", () => {
    const state = useDraftStore.getState();
    const k = keyFor(state.actions, state.previousGames);
    record(k, 50);
    state.configure({ patch: "2" });
    expect(useDraftStore.getState().winRateHistory).toHaveLength(0);
  });

  it("clears history on reset", () => {
    const state = useDraftStore.getState();
    const k = keyFor(state.actions, state.previousGames);
    record(k, 50);
    state.reset();
    expect(useDraftStore.getState().winRateHistory).toHaveLength(0);
  });

  it("clears history on finish", () => {
    const state = useDraftStore.getState();
    const k = keyFor(state.actions, state.previousGames);
    record(k, 50);
    completeDraft();
    state.finish("win");
    expect(useDraftStore.getState().winRateHistory).toHaveLength(0);
  });

  it("rejects old key after finish", () => {
    const state = useDraftStore.getState();
    const k = keyFor(state.actions, state.previousGames);
    record(k, 50);
    completeDraft();
    state.finish("win");
    record(k, 60);
    expect(useDraftStore.getState().winRateHistory).toHaveLength(0);
  });

  it("allows recording after finish with a new key", () => {
    const state = useDraftStore.getState();
    completeDraft();
    state.finish("win");
    const newState = useDraftStore.getState();
    const k = keyFor(newState.actions, newState.previousGames);
    record(k, 48);
    expect(useDraftStore.getState().winRateHistory).toHaveLength(1);
  });

  it("leaves history unchanged by swapPicks", () => {
    const state = useDraftStore.getState();
    const k = keyFor(state.actions, state.previousGames);
    record(k, 50);
    completeDraft();
    state.swapPicks(6, 9);
    expect(useDraftStore.getState().winRateHistory).toEqual([
      { key: k, depth: 0, winRate: 50 },
    ]);
  });
});

describe("draftStateKey", () => {
  it("is deterministic for the same state", () => {
    const key1 = draftStateKey(["a", "b"], []);
    const key2 = draftStateKey(["a", "b"], []);
    expect(key1).toBe(key2);
  });

  it("differs when actions change", () => {
    expect(draftStateKey(["a"], [])).not.toBe(draftStateKey(["b"], []));
  });

  it("differs when previous games change", () => {
    const games = [{ ourPicks: ["a"], opponentPicks: ["b"], result: "win" as const }];
    expect(draftStateKey([], [])).not.toBe(draftStateKey([], games));
  });
});
