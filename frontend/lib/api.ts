import type { components } from "./api-types";

type Schemas = components["schemas"];

/** Reference data used to configure a draft. */
export interface ReferenceData {
  teams: Schemas["TeamSummary"][];
  champions: Schemas["ChampionSummary"][];
  patches: string[];
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) throw new Error(`Draft API returned ${response.status}`);
  return response.json() as Promise<T>;
}

/** Fetch the API's available opponents, champions and patches. */
export async function loadReferenceData(signal?: AbortSignal): Promise<ReferenceData> {
  const [teams, champions, patches] = await Promise.all([
    request<ReferenceData["teams"]>("/teams", { signal }),
    request<ReferenceData["champions"]>("/champions", { signal }),
    request<ReferenceData["patches"]>("/patches", { signal }),
  ]);
  if (!teams.length || !champions.length || !patches.length) {
    throw new Error("Draft API returned incomplete reference data");
  }
  return { teams, champions, patches };
}

/** Load historical rosters independently from draft reference data. */
export function loadRosters(
  opponentTeamId: string,
  signal?: AbortSignal,
): Promise<Schemas["RosterResponse"]> {
  return request(`/rosters?opponentTeamId=${encodeURIComponent(opponentTeamId)}`, { signal });
}

/** Submit a complete draft state to the stateless recommendation API. */
export function recommend(
  draft: Schemas["DraftRequest"],
  signal?: AbortSignal,
): Promise<Schemas["DraftResponse"]> {
  return request("/draft/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(draft),
    signal,
  });
}
