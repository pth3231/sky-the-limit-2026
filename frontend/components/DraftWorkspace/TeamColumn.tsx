import type { components } from "../../lib/api-types";

type Side = "blue" | "red";
type Slot = { index: number; side: Side; action: "pick" | "ban" };
export type HistoricalRoster = components["schemas"]["TeamRoster"];

const ROLES = ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"] as const;
type Role = (typeof ROLES)[number];

const ROLE_ICONS: Record<Role, string> = {
  TOP: "⬆",
  JUNGLE: "🌲",
  MID: "◆",
  BOT: "⬇",
  SUPPORT: "♥",
};

/** Player avatar with role icon */
function PlayerAvatar({
  name,
  role,
  isActive,
}: {
  name: string;
  role?: Role;
  isActive?: boolean;
}) {
  const initials = name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);
  return (
    <div className={`player-avatar ${isActive ? "player-avatar-active" : ""}`}>
      <span className="avatar-initials">{initials}</span>
      {role && <span className="role-icon" aria-label={role}>{ROLE_ICONS[role]}</span>}
    </div>
  );
}

/** Champion portrait in a pick/ban slot */
function SlotPortrait({
  championId,
  names,
  isBan,
}: {
  championId?: string;
  names: Map<string, string>;
  isBan: boolean;
}) {
  if (!championId) return null;
  const name = names.get(championId) ?? championId;
  return (
    <div className={`slot-portrait ${isBan ? "slot-portrait-banned" : ""}`}>
      <span className="portrait-fallback">{name.slice(0, 2).toUpperCase()}</span>
      <span className="slot-name">{name}</span>
    </div>
  );
}

export function TeamColumn({
  name,
  side,
  ours,
  roster,
  order,
  actions,
  names,
  canDraft,
}: {
  name: string;
  side: Side;
  ours: boolean;
  roster?: HistoricalRoster | null;
  order: Slot[];
  actions: string[];
  names: Map<string, string>;
  displayOrder?: number[];
  canDraft: boolean;
  onSwap?: (first: number, second: number) => void;
}) {
  const slots = order.filter((slot) => slot.side === side);
  const picks = slots.filter((item) => item.action === "pick");
  const bans = slots.filter((item) => item.action === "ban");

  // Build player list from roster or generate placeholders
  const players: { name: string; role?: Role }[] =
    roster && roster.players && roster.players.length > 0
      ? roster.players.map((p, i) => ({
          name: p.name ?? `Player ${i + 1}`,
          role: p.role ? (p.role.toUpperCase() as Role) : undefined,
        }))
      : ROLES.map((role, i) => ({ name: `Player ${i + 1}`, role }));

  const activeIndex = canDraft ? actions.length : -1;

  const pickSlotItem = (item: Slot, playerIndex: number) => {
    const active = item.index === activeIndex;
    const id = actions[item.index];
    const player = players[playerIndex % players.length];
    return (
      <li
        key={item.index}
        className={`team-slot team-slot-pick ${active ? "team-slot-active" : ""}`}
        data-draft-action={item.action}
        aria-current={active ? "step" : undefined}
      >
        <PlayerAvatar name={player.name} role={player.role} isActive={active} />
        <SlotPortrait championId={id} names={names} isBan={false} />
        {active && !id && <span className="slot-placeholder">Next pick</span>}
      </li>
    );
  };

  const banSlotItem = (item: Slot, playerIndex: number) => {
    const active = item.index === activeIndex;
    const id = actions[item.index];
    const player = players[playerIndex % players.length];
    return (
      <li
        key={item.index}
        className={`team-slot team-slot-ban ${active ? "team-slot-active" : ""}`}
        data-draft-action={item.action}
        aria-current={active ? "step" : undefined}
      >
        <PlayerAvatar name={player.name} role={player.role} isActive={active} />
        <SlotPortrait championId={id} names={names} isBan={true} />
        {active && !id && <span className="slot-placeholder">Next ban</span>}
      </li>
    );
  };

  return (
    <section
      className={`team-column ${ours ? "team-ours" : "team-opponent"}`}
      aria-label={`${name} draft`}
    >
      <h2 className="team-name">{name}</h2>
      
      {/* Player avatars row */}
      <div className="player-avatars">
        {players.map((player, i) => (
          <PlayerAvatar key={i} name={player.name} role={player.role} />
        ))}
      </div>

      {/* Picks section */}
      <div className="draft-section">
        <h3 className="section-label">PICKS</h3>
        <ol className="team-picks">
          {picks.map((item, i) => pickSlotItem(item, i))}
        </ol>
      </div>

      {/* Bans section */}
      <div className="draft-section">
        <h3 className="section-label">BANS</h3>
        <ol className="team-bans">
          {bans.map((item, i) => banSlotItem(item, i))}
        </ol>
      </div>
    </section>
  );
}
