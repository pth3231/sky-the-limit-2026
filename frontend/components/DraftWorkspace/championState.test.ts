import { describe, expect, it } from "vitest";

import { tileState, tooltipContent, type Candidate } from "./championState";

const candidate = (score: number, reason = "", labels: string[] = []): Candidate => ({
  championId: "a",
  championName: "Alpha",
  score,
  reason,
  contributingFactors: labels.map((label, i) => ({ label, impact: i + 1, signal: "some" })),
});
const base = { banned: false, unavailable: false };

describe("tileState", () => {
  it("lets banned and unavailable override advice", () => {
    expect(tileState({ ...base, banned: true, candidate: candidate(0.9), action: "pick" })).toBe(
      "banned",
    );
    expect(
      tileState({ ...base, unavailable: true, candidate: candidate(0.9), action: "pick" }),
    ).toBe("unavailable");
  });
  it("uses score thresholds for picks", () => {
    expect(tileState({ ...base, candidate: candidate(0.6), action: "pick" })).toBe("recommended");
    expect(tileState({ ...base, candidate: candidate(0.4), action: "pick" })).toBe("discouraged");
    expect(tileState({ ...base, candidate: candidate(0.5), action: "pick" })).toBe("neutral");
  });
  it("highlights only the top ranked ban", () => {
    expect(
      tileState({ ...base, candidate: candidate(0.1), action: "ban", topCandidate: true }),
    ).toBe("recommended");
    expect(tileState({ ...base, candidate: candidate(0.9), action: "ban" })).toBe("neutral");
  });
  it("is neutral without candidate or action", () => {
    expect(tileState({ ...base })).toBe("neutral");
  });
});

describe("tooltipContent", () => {
  it("builds a template from the top factors when no reason exists", () => {
    const { headline, body } = tooltipContent(
      candidate(0.62, "  ", ["small", "big", "mid"]),
      "recommended",
      "pick",
    );
    expect(headline).toContain("62%");
    expect(body).toBe("Alpha looks like a strong pick here: mid; big.");
  });
  it("replaces the template with the server reason", () => {
    expect(tooltipContent(candidate(0.4, "AI says no"), "discouraged", "pick").body).toBe(
      "AI says no",
    );
  });
  it("does not call a ban score a win rate", () => {
    expect(tooltipContent(candidate(0.62), "recommended", "ban").headline).toBe(
      "Ban candidate score: 0.62",
    );
  });
  it("does not call a neutral candidate strong", () => {
    expect(tooltipContent(candidate(0.5), "neutral", "pick").body).not.toContain("strong");
  });
});
