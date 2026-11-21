import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DraftWorkspace, type DraftWorkspaceProps } from "./DraftWorkspace";
import { DRAFT_ORDER } from "../../lib/draft";

afterEach(cleanup);

const order: DraftWorkspaceProps["order"] = Array.from({ length: 20 }, (_, index) => ({
  index,
  side: index % 2 === 0 ? "blue" : "red",
  action: index < 6 ? "ban" : "pick",
}));

function props(overrides: Partial<DraftWorkspaceProps> = {}): DraftWorkspaceProps {
  return {
    setup: { opponentTeamId: "team", seriesFormat: "bo3", patch: "1.0", ourSide: "blue" },
    onSetupChange: vi.fn(),
    teams: [
      { teamId: "team", name: "Rivals" },
      { teamId: "other", name: "Challengers" },
    ],
    champions: [
      { championId: "a", name: "Alpha" },
      { championId: "b", name: "Bravo" },
      { championId: "c", name: "Charlie" },
    ],
    patches: ["1.0", "2.0"],
    dataStatus: "ready",
    onPractice: vi.fn(),
    onRetry: vi.fn(),
    actions: [],
    previousGames: [],
    order,
    fearlessExcluded: [],
    seriesDecided: false,
    onLock: vi.fn(),
    onUndo: vi.fn(),
    onFinish: vi.fn(),
    recommendation: null,
    recommendationStatus: "idle",
    recommendationError: "",
    ...overrides,
  };
}

describe("DraftWorkspace", () => {
  it("shows both teams' picks and bans in the draft bar", () => {
    render(<DraftWorkspace {...props({ order: DRAFT_ORDER, actions: ["a"] })} />);
    const draftBar = screen.getByRole("region", { name: /Draft status/i });
    expect(draftBar.querySelectorAll('[data-draft-action="pick"]')).toHaveLength(10);
    expect(draftBar.querySelectorAll('[data-draft-action="ban"]')).toHaveLength(10);
    const activeSlot = draftBar.querySelector('.draft-slot-active');
    expect(activeSlot).toHaveAttribute("aria-current", "step");
  });

  it("lays out a header, selector, stats panel, and draft bar in one screen", () => {
    const { container } = render(<DraftWorkspace {...props()} />);
    const header = container.querySelector(".draft-header");
    expect(header).toBeInTheDocument();
    expect(screen.getByText(/BAN A CHAMPION!/i)).toBeInTheDocument();
    const screenEl = container.querySelector(".draft-screen") as HTMLElement;
    expect(screenEl.querySelector(".pool-section")).toBeInTheDocument();
    expect(screenEl.querySelector(".stats-panel")).toBeInTheDocument();
    const footer = container.querySelector(".draft-footer");
    expect(footer).toBeInTheDocument();
  });

  it("shows the current action label and active team", () => {
    render(<DraftWorkspace {...props({ order: DRAFT_ORDER, actions: ["a"] })} />);
    // Header title is "Draft Recommendation"; subtitle shows action and team
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/Draft Recommendation/i);
    const subtitle = screen.getByText(/BAN A CHAMPION!/i);
    expect(subtitle).toHaveTextContent(/Rivals/);
  });

  it("shows draft complete and series decided states", () => {
    const input = props();
    const { rerender } = render(<DraftWorkspace {...input} />);
    rerender(<DraftWorkspace {...input} actions={Array.from({ length: 20 }, (_, i) => `${i}`)} />);
    // The header title is always "Draft Recommendation"; status appears in the subtitle
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/Draft Recommendation/i);
    expect(screen.getByText(/Draft complete/i)).toBeInTheDocument();
    rerender(<DraftWorkspace {...input} seriesDecided />);
    expect(screen.getByText(/Series decided/i)).toBeInTheDocument();
  });

  it("locks only available champions, supports search and undo", () => {
    const input = props({ actions: ["a"], fearlessExcluded: ["b"] });
    render(<DraftWorkspace {...input} />);
    expect(screen.getByRole("button", { name: /alpha/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /bravo/i })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /charlie/i }));
    expect(input.onLock).toHaveBeenCalledWith("c");
    fireEvent.click(screen.getByRole("button", { name: /undo/i }));
    expect(input.onUndo).toHaveBeenCalledOnce();
  });

  it("styles recommended and discouraged tiles with focus-linked tooltips", () => {
    const cand = (championId: string, championName: string, score: number, reason: string) => ({
      championId,
      championName,
      score,
      reason,
      contributingFactors: [],
    });
    const base = props();
    const rec = {
      gameNumber: 1,
      patchesUsed: [],
      nextSlot: { index: 6, side: "blue", action: "pick" },
      recommendation: cand("b", "Bravo", 0.6, "Strong into their comp."),
      alternatives: [cand("c", "Charlie", 0.4, "")],
      winRate: 0.5,
      gameWinRate: 0.5,
      opponentTendencies: [],
      contributingFactors: [],
      fearlessExcludedIds: [],
      unavailableIds: [],
      sampleNote: null,
      engine: "statistical",
    } as unknown as DraftWorkspaceProps["recommendation"];
    render(
      <DraftWorkspace
        {...base}
        dataStatus="ready"
        recommendationStatus="ready"
        recommendation={rec}
        actions={["a"]}
        order={Array.from({ length: 20 }, (_, index) => ({
          index,
          side: "blue" as const,
          action: index === 0 ? ("ban" as const) : ("pick" as const),
        }))}
      />,
    );
    const bravo = screen.getByRole("button", { name: /bravo/i });
    expect(bravo).toHaveClass("champion-recommended");
    expect(screen.getByRole("button", { name: /charlie/i })).toHaveClass("champion-discouraged");
    expect(screen.getByRole("button", { name: /alpha/i })).toHaveClass("champion-banned");
    const tip = document.getElementById(bravo.getAttribute("aria-describedby") as string);
    expect(tip).toHaveAttribute("role", "tooltip");
    expect(tip).toHaveTextContent("Strong into their comp.");
  });

  it("explains a neutral scored candidate and keeps its tooltip open while focused", () => {
    const rec = {
      gameNumber: 1,
      patchesUsed: [],
      nextSlot: { index: 6, side: "blue", action: "pick" },
      recommendation: {
        championId: "b",
        championName: "Bravo",
        score: 0.5,
        reason: "Even matchup",
        contributingFactors: [],
      },
      alternatives: [],
      winRate: 0.5,
      gameWinRate: 0.5,
      opponentTendencies: [],
      contributingFactors: [],
      fearlessExcludedIds: [],
      unavailableIds: [],
      sampleNote: null,
      engine: "statistical",
    } as unknown as DraftWorkspaceProps["recommendation"];
    render(
      <DraftWorkspace
        {...props()}
        dataStatus="ready"
        recommendationStatus="ready"
        recommendation={rec}
        actions={["a"]}
        order={Array.from({ length: 20 }, (_, index) => ({
          index,
          side: "blue" as const,
          action: index === 0 ? ("ban" as const) : ("pick" as const),
        }))}
      />,
    );
    const button = screen.getByRole("button", { name: /bravo/i });
    fireEvent.focus(button);
    const tip = document.getElementById(button.getAttribute("aria-describedby") as string);
    expect(tip).toHaveTextContent("Even matchup");
    expect(tip).toHaveAttribute("data-open", "true");
    fireEvent.mouseLeave(button.parentElement!);
    expect(tip).toHaveAttribute("data-open", "true");
  });

  it("loads versioned champion portraits and falls back when an image fails", () => {
    const input = props({
      champions: [
        {
          championId: "a",
          name: "Aatrox",
          role: null,
          portraitUrl: "https://ddragon.leagueoflegends.com/cdn/16.1.1/img/champion/Aatrox.png",
        },
        { championId: "b", name: "Ahri", role: null },
      ],
    });
    render(<DraftWorkspace {...input} />);

    const portrait = document.querySelector(".champion-portrait img") as HTMLImageElement;
    expect(portrait.src).toBe(
      "https://ddragon.leagueoflegends.com/cdn/16.1.1/img/champion/Aatrox.png",
    );
    expect(portrait).toHaveAttribute("alt", "");
    expect(screen.getByText("AH")).toBeInTheDocument();
    fireEvent.error(portrait);
    expect(document.querySelector(".champion-portrait img")).not.toBeInTheDocument();
    expect(screen.getByText("AA")).toBeInTheDocument();
    expect(input.onLock).not.toHaveBeenCalled();
  });

  it("does not lock banned or fearless-excluded champions", () => {
    const onLock = vi.fn();
    render(
      <DraftWorkspace
        {...props()}
        onLock={onLock}
        actions={["a"]}
        fearlessExcluded={["b"]}
        order={Array.from({ length: 20 }, (_, index) => ({
          index,
          side: "blue" as const,
          action: index === 0 ? ("ban" as const) : ("pick" as const),
        }))}
      />,
    );
    expect(screen.getByRole("button", { name: /alpha/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /bravo/i })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /bravo/i }));
    expect(onLock).not.toHaveBeenCalled();
  });

  it("shows the tooltip on focus and hides it on blur or Escape, outside the scroll grid", () => {
    const cand = {
      championId: "b",
      championName: "Bravo",
      score: 0.6,
      reason: "",
      contributingFactors: [{ label: "Wins lane", impact: 0.03, signal: "some" }],
    };
    const rec = {
      gameNumber: 1,
      patchesUsed: [],
      nextSlot: { index: 6, side: "blue", action: "pick" },
      recommendation: cand,
      alternatives: [],
      winRate: 0.5,
      gameWinRate: 0.5,
      opponentTendencies: [],
      contributingFactors: [],
      fearlessExcludedIds: [],
      unavailableIds: [],
      sampleNote: null,
      engine: "statistical",
    } as unknown as DraftWorkspaceProps["recommendation"];
    render(
      <DraftWorkspace
        {...props()}
        dataStatus="ready"
        recommendationStatus="ready"
        recommendation={rec}
        actions={["a"]}
        order={Array.from({ length: 20 }, (_, index) => ({
          index,
          side: "blue" as const,
          action: index === 0 ? ("ban" as const) : ("pick" as const),
        }))}
      />,
    );
    const bravo = screen.getByRole("button", { name: /bravo/i });
    const tip = document.getElementById(bravo.getAttribute("aria-describedby") as string);
    expect(tip).toHaveAttribute("data-open", "false");
    expect(tip?.closest(".champion-grid")).toBeNull();
    fireEvent.focus(bravo);
    expect(tip).toHaveAttribute("data-open", "true");
    expect(tip).toHaveTextContent("Wins lane");
    fireEvent.keyDown(window, { key: "Escape" });
    expect(tip).toHaveAttribute("data-open", "false");
    fireEvent.focus(bravo);
    fireEvent.blur(bravo);
    expect(tip).toHaveAttribute("data-open", "false");
  });

  it("shows an empty message when search matches nothing", () => {
    render(<DraftWorkspace {...props()} />);
    fireEvent.change(screen.getByPlaceholderText(/search champions/i), { target: { value: "zzz" } });
    expect(screen.getByText("No champions found.")).toBeInTheDocument();
  });

  it("only offers results for a completed draft and prevents another decided game", () => {
    const input = props();
    const { rerender } = render(<DraftWorkspace {...input} />);
    expect(screen.queryByRole("button", { name: /record win/i })).not.toBeInTheDocument();
    rerender(<DraftWorkspace {...input} actions={Array.from({ length: 20 }, (_, i) => `${i}`)} />);
    fireEvent.click(screen.getByRole("button", { name: /record win/i }));
    fireEvent.click(screen.getByRole("button", { name: /record loss/i }));
    expect(input.onFinish).toHaveBeenCalledWith("win");
    expect(input.onFinish).toHaveBeenCalledWith("loss");
    rerender(<DraftWorkspace {...input} seriesDecided />);
    expect(screen.getByText(/SERIES DECIDED/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /alpha/i })).toBeDisabled();
    expect(screen.queryByRole("button", { name: /record win/i })).not.toBeInTheDocument();
  });

  it("shows undo button in footer when draft is active", () => {
    const input = props();
    render(<DraftWorkspace {...input} />);
    const undoButton = screen.getByRole("button", { name: /undo/i });
    expect(undoButton).toBeInTheDocument();
    expect(undoButton).toHaveClass("undo-button");
  });

  it("shows finish actions when draft is complete", () => {
    const input = props();
    const { rerender } = render(<DraftWorkspace {...input} />);
    expect(screen.queryByText(/how did cloud9 finish/i)).not.toBeInTheDocument();
    rerender(<DraftWorkspace {...input} actions={Array.from({ length: 20 }, (_, i) => `${i}`)} />);
    expect(screen.getByText(/how did cloud9 finish/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /record win/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /record loss/i })).toBeInTheDocument();
  });
});
