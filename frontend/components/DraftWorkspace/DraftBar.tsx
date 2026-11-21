import type { components } from "../../lib/api-types";
import type { Side } from "./TopBar";

type Result = "win" | "loss";
type Slot = { index: number; side: Side; action: "ban" | "pick" };

export interface DraftBarProps {
  ourTeam: string;
  opponentTeam: string;
  order: Slot[];
  actions: string[];
  champions: { championId: string; name: string; portraitUrl?: string | null }[];
  rosters?: { ourTeam: components["schemas"]["TeamRoster"]; opponentTeam: components["schemas"]["TeamRoster"] } | null;
  canDraft: boolean;
  seriesDecided: boolean;
  complete: boolean;
  onUndo(): void;
  onFinish(result: Result): void;
  onSwapPicks?(first: number, second: number): void;
  displayOrder?: number[];
}

/** Bottom draft bar showing both teams' bans, picks, and player names. */
export function DraftBar({
  ourTeam,
  opponentTeam,
  order,
  actions,
  champions,
  rosters = null,
  canDraft,
  seriesDecided,
  complete,
  onUndo,
  onFinish,
}: DraftBarProps) {
  const names = new Map(champions.map((c) => [c.championId, c.name]));
  const portraits = new Map(champions.map((c) => [c.championId, c.portraitUrl]));
  const ourSlots = order.filter((s) => s.side === "blue");
  const theirSlots = order.filter((s) => s.side === "red");
  const activeIndex = canDraft ? actions.length : -1;

  // Build player names from rosters or placeholders
  const ourPlayers = rosters?.ourTeam?.players
    ? rosters.ourTeam.players.map((p) => p.name ?? "Player")
    : ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"];
  const theirPlayers = rosters?.opponentTeam?.players
    ? rosters.opponentTeam.players.map((p) => p.name ?? "Player")
    : ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"];

  return (
    <footer className="draft-footer">
      {/* Undo button */}
      {!seriesDecided && (
        <button
          className="button button-outline undo-button"
          onClick={onUndo}
          disabled={actions.length === 0 || !canDraft}
          aria-label="Undo last action"
        >
          ↶ Undo
        </button>
      )}

      {/* Draft bar - both teams */}
      <div className="draft-bar-main" role="region" aria-label="Draft status">
        {/* Our team - left side */}
        <div className="draft-bar-team draft-bar-ours">
          <div className="team-logo-placeholder" aria-label={`${ourTeam} logo`}>
            <span className="team-logo-initials">{ourTeam.slice(0, 2).toUpperCase()}</span>
          </div>
          <div className="team-info">
            <span className="team-label">{ourTeam}</span>
            <div className="draft-bar-slots">
            {/* Bans */}
            {ourSlots.filter((s) => s.action === "ban").map((slot, i) => (
              <DraftSlot
                key={slot.index}
                slot={slot}
                championId={actions[slot.index]}
                name={names.get(actions[slot.index]) ?? ""}
                portraitUrl={portraits.get(actions[slot.index])}
                playerName={ourPlayers[i] ?? ""}
                isActive={slot.index === activeIndex}
                isBan
              />
            ))}
            {/* Picks */}
            {ourSlots.filter((s) => s.action === "pick").map((slot, i) => (
              <DraftSlot
                key={slot.index}
                slot={slot}
                championId={actions[slot.index]}
                name={names.get(actions[slot.index]) ?? ""}
                portraitUrl={portraits.get(actions[slot.index])}
                playerName={ourPlayers[i] ?? ""}
                isActive={slot.index === activeIndex}
                isBan={false}
              />
            ))}
            </div>
          </div>
        </div>

        {/* VS divider */}
        <div className="draft-bar-divider">
          <span className="vs-label">VS</span>
        </div>

        {/* Opponent team - right side */}
        <div className="draft-bar-team draft-bar-theirs">
          <div className="team-logo-placeholder" aria-label={`${opponentTeam} logo`}>
            <span className="team-logo-initials">{opponentTeam.slice(0, 2).toUpperCase()}</span>
          </div>
          <div className="team-info">
            <span className="team-label">{opponentTeam}</span>
            <div className="draft-bar-slots">
            {/* Bans */}
            {theirSlots.filter((s) => s.action === "ban").map((slot, i) => (
              <DraftSlot
                key={slot.index}
                slot={slot}
                championId={actions[slot.index]}
                name={names.get(actions[slot.index]) ?? ""}
                portraitUrl={portraits.get(actions[slot.index])}
                playerName={theirPlayers[i] ?? ""}
                isActive={slot.index === activeIndex}
                isBan
              />
            ))}
            {/* Picks */}
            {theirSlots.filter((s) => s.action === "pick").map((slot, i) => (
              <DraftSlot
                key={slot.index}
                slot={slot}
                championId={actions[slot.index]}
                name={names.get(actions[slot.index]) ?? ""}
                portraitUrl={portraits.get(actions[slot.index])}
                playerName={theirPlayers[i] ?? ""}
                isActive={slot.index === activeIndex}
                isBan={false}
              />
            ))}
            </div>
          </div>
        </div>
      </div>

      {/* Finish actions */}
      {complete && !seriesDecided && (
        <div className="finish-actions">
          <p>How did Cloud9 finish?</p>
          <button className="button button-primary" onClick={() => onFinish("win")}>
            Record win
          </button>
          <button className="button button-outline" onClick={() => onFinish("loss")}>
            Record loss
          </button>
        </div>
      )}
    </footer>
  );
}

function DraftSlot({
  slot,
  championId,
  name,
  portraitUrl,
  playerName,
  isActive,
  isBan,
}: {
  slot: Slot;
  championId?: string;
  name: string;
  portraitUrl?: string | null;
  playerName: string;
  isActive: boolean;
  isBan: boolean;
}) {
  const showImage = Boolean(portraitUrl) && championId;
  const initials = name.slice(0, 2).toUpperCase();

  return (
    <div
      className={`draft-slot ${isBan ? "draft-slot-ban" : "draft-slot-pick"} ${isActive ? "draft-slot-active" : ""}`}
      data-draft-action={slot.action}
      aria-current={isActive ? "step" : undefined}
    >
      <div className="draft-slot-portrait">
        {championId ? (
          showImage ? (
            <img src={portraitUrl!} alt="" aria-hidden="true" loading="lazy" decoding="async" />
          ) : (
            <span className="portrait-fallback">{initials}</span>
          )
        ) : (
          <span className="slot-empty" aria-label={isBan ? "Ban slot" : "Pick slot"}>
            {isBan ? "🚫" : "?"}
          </span>
        )}
      </div>
      <span className="draft-slot-player">{playerName}</span>
    </div>
  );
}
