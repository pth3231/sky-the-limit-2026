import { tooltipContent, type Candidate, type TileState } from "./championState";
import type { RefObject } from "react";

/** Small pop-up shown on hover or keyboard focus of a tile, anchored to the viewport. */
export function ChampionTooltip({
  id,
  candidate,
  state,
  action,
  open,
  position,
  tooltipRef,
  onMouseEnter,
  onMouseLeave,
}: {
  id: string;
  candidate: Candidate;
  state: TileState;
  action: "ban" | "pick";
  open: boolean;
  position: { left: number; top: number; below: boolean } | null;
  tooltipRef: RefObject<HTMLSpanElement | null>;
  onMouseEnter(): void;
  onMouseLeave(): void;
}) {
  const { headline, body } = tooltipContent(candidate, state, action);
  return (
    <span
      role="tooltip"
      ref={tooltipRef}
      id={id}
      className="champion-tooltip"
      data-open={open ? "true" : "false"}
      data-below={position?.below ? "true" : "false"}
      style={position ? { left: position.left, top: position.top } : undefined}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <strong>{headline}</strong>
      <span>{body}</span>
    </span>
  );
}
