import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useDraftStore } from "./draft";
import { useDraftSession } from "./session";
import { loadReferenceData, loadRosters, recommend } from "./api";

vi.mock("./api", () => ({ loadReferenceData: vi.fn(), loadRosters: vi.fn(), recommend: vi.fn() }));

const reference = {
  teams: [{ teamId: "team", name: "Opponent" }],
  champions: [
    { championId: "one", name: "One" },
    { championId: "two", name: "Two" },
  ],
  patches: ["1.0"],
};

describe("draft session", () => {
  afterEach(cleanup);
  beforeEach(() => {
    useDraftStore.getState().reset();
    vi.resetAllMocks();
    vi.mocked(loadRosters).mockResolvedValue(null as never);
  });

  it("offers explicit synthetic practice when reference data cannot load", async () => {
    vi.mocked(loadReferenceData).mockRejectedValue(new Error("offline"));
    const { result } = renderHook(() => useDraftSession());
    await waitFor(() => expect(result.current.dataStatus).toBe("error"));
    act(() => result.current.onPractice());
    expect(result.current.dataStatus).toBe("practice");
    expect(result.current.champions[0].name).toContain("Practice");
    expect(result.current.recommendation).toBeNull();
    expect(recommend).not.toHaveBeenCalled();
  });

  it("ignores responses for an earlier draft even when requests resolve out of order", async () => {
    vi.mocked(loadReferenceData).mockResolvedValue(reference);
    const pending: ((value: Awaited<ReturnType<typeof recommend>>) => void)[] = [];
    vi.mocked(recommend).mockImplementation(() => new Promise((resolve) => pending.push(resolve)));
    const { result } = renderHook(() => useDraftSession());
    await waitFor(() => expect(pending).toHaveLength(1));
    act(() => result.current.onLock("one"));
    await waitFor(() => expect(pending).toHaveLength(2));
    const response = { winRate: 0.8 } as Awaited<ReturnType<typeof recommend>>;
    await act(async () => pending[1](response));
    expect(result.current.recommendation).toBe(response);
    await act(async () => pending[0]({ winRate: 0.1 } as typeof response));
    expect(result.current.recommendation).toBe(response);
  });

  it("loads rosters independently and ignores stale opponent responses", async () => {
    vi.mocked(loadReferenceData).mockResolvedValue(reference);
    vi.mocked(recommend).mockReturnValue(new Promise(() => {}));
    const pending: ((value: Awaited<ReturnType<typeof loadRosters>>) => void)[] = [];
    vi.mocked(loadRosters).mockImplementation(
      () => new Promise((resolve) => pending.push(resolve)),
    );
    const { result } = renderHook(() => useDraftSession());
    await waitFor(() => expect(pending).toHaveLength(1));
    expect(result.current.dataStatus).toBe("ready");
    act(() => result.current.onSetupChange({ opponentTeamId: "other" }));
    await waitFor(() => expect(pending).toHaveLength(2));
    const team = (teamId: string) => ({ teamId, name: teamId, asOf: null, players: [] });
    const current = { ourTeam: team("ours"), opponentTeam: team("other") };
    await act(async () => pending[1](current));
    expect(result.current.rosters).toEqual(current);
    await act(async () => pending[0]({ ourTeam: team("ours"), opponentTeam: team("team") }));
    expect(result.current.rosters).toEqual(current);
  });
});
