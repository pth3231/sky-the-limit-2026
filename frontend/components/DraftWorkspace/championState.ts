import type { components } from "../../lib/api-types";

export type Candidate = components["schemas"]["Candidate"];
export type TileState = "banned" | "unavailable" | "recommended" | "discouraged" | "neutral";

/** Pick score (projected game win chance) at or above this is highly recommended. */
export const RECOMMENDED_SCORE = 0.55;
/** Pick score at or below this is a bad pick. */
export const DISCOURAGED_SCORE = 0.45;

interface TileInput {
  /** Drafted as a ban in this game. */
  banned: boolean;
  /** Drafted as a pick this game, or used in an earlier fearless game. */
  unavailable: boolean;
  candidate?: Candidate;
  action?: "ban" | "pick";
  topCandidate?: boolean;
}

/** Decide how a champion tile is styled. Ban/unavailable always win over advice. */
export function tileState({
  banned,
  unavailable,
  candidate,
  action,
  topCandidate,
}: TileInput): TileState {
  if (banned) return "banned";
  if (unavailable) return "unavailable";
  if (!candidate || !action) return "neutral";
  if (action === "ban") return topCandidate ? "recommended" : "neutral";
  if (candidate.score >= RECOMMENDED_SCORE) return "recommended";
  if (candidate.score <= DISCOURAGED_SCORE) return "discouraged";
  return "neutral";
}

const percent = (value: number) => `${Number((value * 100).toFixed(1))}%`;

/** Build tooltip content. The server reason replaces the generated template when present. */
export function tooltipContent(
  candidate: Candidate,
  state: TileState,
  action: "ban" | "pick",
): { headline: string; body: string } {
  const headline =
    action === "pick"
      ? `Projected win chance with pick: ${percent(candidate.score)}`
      : `Ban candidate score: ${candidate.score.toFixed(2)}`;
  const factors = [...candidate.contributingFactors]
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .slice(0, 2)
    .map((factor) => factor.label);
  const verdict =
    state === "discouraged"
      ? `${candidate.championName} looks like a weak ${action} here`
      : state === "recommended" && action === "pick"
        ? `${candidate.championName} looks like a strong ${action} here`
        : `${candidate.championName} is a suggested ${action}`;
  const template = factors.length ? `${verdict}: ${factors.join("; ")}.` : `${verdict}.`;
  return { headline, body: candidate.reason.trim() || template };
}
