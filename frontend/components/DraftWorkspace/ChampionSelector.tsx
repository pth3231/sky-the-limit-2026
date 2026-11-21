import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { ChampionTooltip } from "./ChampionTooltip";
import { tileState, type Candidate, type TileState } from "./championState";

export interface ChampionSelectorProps {
  champions: {
    championId: string;
    name: string;
    role?: Role | null;
    portraitUrl?: string | null;
  }[];
  actions: string[];
  /** Action kind per locked slot, aligned with `actions`. */
  slotActions: ("ban" | "pick")[];
  fearlessExcluded: string[];
  candidates: Candidate[];
  currentAction?: "ban" | "pick";
  locked: boolean;
  onLock(id: string): void;
}

export type Role = "top" | "jungle" | "mid" | "bot" | "support";

export const ROLES: { role: Role; label: string }[] = [
  { role: "top", label: "Top" },
  { role: "jungle", label: "Jungle" },
  { role: "mid", label: "Mid" },
  { role: "bot", label: "Bot" },
  { role: "support", label: "Support" },
];

const ROLE_MARKER: Record<Role, { x: number; y: number }> = {
  top: { x: 7, y: 6 },
  jungle: { x: 16, y: 10 },
  mid: { x: 12, y: 12 },
  bot: { x: 17, y: 18 },
  support: { x: 14, y: 18 },
};

function RoleIcon({ role }: { role: Role }) {
  const { x, y } = ROLE_MARKER[role];
  return (
    <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
      <path d="M4 20 20 4M4 4l16 16M5 5h5M19 19h-5" />
      <circle cx={x} cy={y} r="3" />
      <circle cx={x} cy={y} r="1" className="role-marker-center" />
    </svg>
  );
}

/** Whether a champion passes the role filter. Unknown-role champions only show with no filter. */
export function matchesRole(role: Role | null | undefined, filter: Role | null): boolean {
  return filter === null || role === filter;
}

const STATE_LABEL: Record<TileState, string> = {
  banned: "Banned",
  unavailable: "Unavailable",
  recommended: "Recommended",
  discouraged: "Discouraged",
  neutral: "Available",
};

const TAG: Record<TileState, string> = {
  banned: "BANNED",
  unavailable: "USED",
  recommended: "FAVORED",
  discouraged: "RISKY",
  neutral: "",
};

interface TileProps {
  id: string;
  name: string;
  portraitUrl?: string | null;
  state: TileState;
  fearless: boolean;
  disabled: boolean;
  candidate?: Candidate;
  action?: "ban" | "pick";
  onLock(id: string): void;
}

const ChampionTile = memo(function ChampionTile({
  id,
  name,
  portraitUrl,
  state,
  fearless,
  disabled,
  candidate,
  action,
  onLock,
}: TileProps) {
  const tipId = `tip-${id}`;
  const cellRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLSpanElement>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number; below: boolean } | null>(null);
  const place = useCallback(() => {
    const rect = cellRef.current?.getBoundingClientRect();
    if (!rect) return;
    const tip = tooltipRef.current?.getBoundingClientRect();
    const halfWidth = (tip?.width || 220) / 2;
    const below = rect.top < (tip?.height || 150) + 6;
    const left = Math.min(
      Math.max(rect.left + rect.width / 2, halfWidth),
      window.innerWidth - halfWidth,
    );
    setPos({ left, top: below ? rect.bottom + 6 : rect.top - 6, below });
  }, []);
  useLayoutEffect(() => {
    if (open) place();
  }, [open, place]);
  useEffect(() => {
    if (!open) return;
    const close = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
      window.removeEventListener("keydown", close);
    };
  }, [open, place]);
  useEffect(
    () => () => {
      if (closeTimer.current) clearTimeout(closeTimer.current);
    },
    [],
  );
  const showTip = candidate && action && !disabled;
  const show = () => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    setOpen(true);
  };
  const close = () => {
    if (cellRef.current?.contains(document.activeElement)) return;
    closeTimer.current = setTimeout(() => setOpen(false), 150);
  };
  const label = fearless ? "Fearless excluded" : STATE_LABEL[state];
  return (
    <div
      className="champion-cell"
      ref={cellRef}
      onMouseEnter={show}
      onMouseLeave={close}
      onFocus={show}
      onBlur={() => setOpen(false)}
    >
      <button
        className={`champion champion-${state}`}
        disabled={disabled}
        onClick={() => onLock(id)}
        aria-label={`${name} — ${fearless ? label : state === "neutral" ? "Available" : label}`}
        aria-describedby={showTip ? tipId : undefined}
      >
        <ChampionPortrait name={name} portraitUrl={portraitUrl} />
        <span className="champion-name">{name}</span>
        <span className="champion-tag" aria-hidden="true">
          {fearless ? "FEARLESS" : TAG[state]}
        </span>
      </button>
      {showTip &&
        typeof document !== "undefined" &&
        createPortal(
          <ChampionTooltip
            id={tipId}
            candidate={candidate}
            state={state}
            action={action}
            open={open}
            position={pos}
            tooltipRef={tooltipRef}
            onMouseEnter={show}
            onMouseLeave={close}
          />,
          document.body,
        )}
    </div>
  );
});

function ChampionPortrait({ name, portraitUrl }: { name: string; portraitUrl?: string | null }) {
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const showImage = Boolean(portraitUrl) && failedUrl !== portraitUrl;

  return (
    <span className="champion-portrait" aria-hidden="true">
      {showImage ? (
        <img
          src={portraitUrl!}
          alt=""
          loading="lazy"
          decoding="async"
          onError={() => setFailedUrl(portraitUrl ?? null)}
        />
      ) : (
        <span className="portrait-fallback">{name.slice(0, 2).toUpperCase()}</span>
      )}
    </span>
  );
}

/** Search plus a grid of champion tiles. Only the grid scrolls. */
export function ChampionSelector({
  champions,
  actions,
  slotActions,
  fearlessExcluded,
  candidates,
  currentAction,
  locked,
  onLock,
}: ChampionSelectorProps) {
  const [query, setQuery] = useState("");
  const [roleFilter, setRoleFilter] = useState<Role | null>(null);
  const [sortBy, setSortBy] = useState<"name" | "role">("name");
  const bans = useMemo(
    () => new Set(actions.filter((_, i) => slotActions[i] === "ban")),
    [actions, slotActions],
  );
  const used = useMemo(() => new Set(actions), [actions]);
  const fearless = useMemo(() => new Set(fearlessExcluded), [fearlessExcluded]);
  const byId = useMemo(() => new Map(candidates.map((c) => [c.championId, c])), [candidates]);
  const needle = query.trim().toLocaleLowerCase();
  const visible = useMemo(() => {
    let filtered = champions.filter(
      (c) => c.name.toLocaleLowerCase().includes(needle),
    );
    // Apply role filter
    if (roleFilter) {
      filtered = filtered.filter((c) => matchesRole(c.role, roleFilter));
    }
    // Apply sorting
    if (sortBy === "name") {
      filtered = [...filtered].sort((a, b) => a.name.localeCompare(b.name));
    } else if (sortBy === "role") {
      const roleOrder: Record<string, number> = { top: 0, jungle: 1, mid: 2, bot: 3, support: 4 };
      filtered = [...filtered].sort((a, b) => {
        const ra = roleOrder[a.role ?? ""] ?? 99;
        const rb = roleOrder[b.role ?? ""] ?? 99;
        return ra !== rb ? ra - rb : a.name.localeCompare(b.name);
      });
    }
    return filtered;
  }, [champions, needle, roleFilter, sortBy]);

  return (
    <>
      <div className="pool-toolbar">
        <input
          id="champion-search"
          type="search"
          placeholder="Search champions"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <select
          className="sort-select"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as "name" | "role")}
          aria-label="Sort champions by"
        >
          <option value="name">Sort by Name</option>
          <option value="role">Sort by Role</option>
        </select>
        <div className="role-filters" role="group" aria-label="Filter by role">
          {ROLES.map((r) => (
            <button
              key={r.role}
              type="button"
              className={`role-button ${roleFilter === r.role ? "role-button-active" : ""}`}
              aria-pressed={roleFilter === r.role}
              aria-label={`Filter by ${r.label}`}
              onClick={() => setRoleFilter(roleFilter === r.role ? null : r.role)}
              title={r.label}
            >
              <RoleIcon role={r.role} />
            </button>
          ))}
        </div>
      </div>
      <div className="champion-grid" role="region" aria-label="Champion pool" tabIndex={0}>
        {visible.map((champion) => {
          const id = champion.championId;
          const isFearless = fearless.has(id);
          const candidate = byId.get(id);
          const state = tileState({
            banned: bans.has(id),
            unavailable: used.has(id) || isFearless,
            candidate,
            action: currentAction,
            topCandidate: candidates[0]?.championId === id,
          });
          return (
            <ChampionTile
              key={id}
              id={id}
              name={champion.name}
              portraitUrl={champion.portraitUrl}
              state={state}
              fearless={isFearless && !used.has(id)}
              disabled={used.has(id) || isFearless || locked}
              candidate={candidate}
              action={currentAction}
              onLock={onLock}
            />
          );
        })}
      </div>
      {visible.length === 0 && <p className="empty-message">No champions found.</p>}
    </>
  );
}
