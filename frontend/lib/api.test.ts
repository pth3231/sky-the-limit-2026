import { afterEach, describe, expect, it, vi } from "vitest";
import { loadReferenceData, recommend } from "./api";

describe("API client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("loads all reference data through the same-origin proxy", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      const data = url.endsWith("teams") ? [{ teamId: "t", name: "Team" }] :
        url.endsWith("champions") ? [{ championId: "c", name: "Champion" }] : ["1.0"];
      return new Response(JSON.stringify(data), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    expect(await loadReferenceData()).toEqual({
      teams: [{ teamId: "t", name: "Team" }],
      champions: [{ championId: "c", name: "Champion" }],
      patches: ["1.0"],
    });
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/api/teams", "/api/champions", "/api/patches",
    ]);
  });

  it("reports backend errors instead of fabricating data", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("error", { status: 503 })));
    await expect(
      recommend({ opponentTeamId: "t", ourSide: "blue", seriesFormat: "bo3" }),
    ).rejects.toThrow("503");
  });
});