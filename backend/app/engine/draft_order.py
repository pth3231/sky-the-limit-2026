"""The standard 20-slot professional draft order."""

from pydantic import BaseModel, ConfigDict

from app.engine.models import ActionType, Side, Slot, SlotOwner


class DraftSlot(BaseModel):
    """A position in the draft order and the side and action it belongs to."""

    model_config = ConfigDict(frozen=True)

    index: int
    side: Side
    action: ActionType


_PHASES: tuple[tuple[ActionType, str], ...] = (
    ("ban", "BRBRBR"),
    ("pick", "BRRBBR"),
    ("ban", "RBRB"),
    ("pick", "RBBR"),
)


def _build_order() -> tuple[DraftSlot, ...]:
    sides: dict[str, Side] = {"B": "blue", "R": "red"}
    slots: list[DraftSlot] = []
    for action, pattern in _PHASES:
        for letter in pattern:
            slots.append(DraftSlot(index=len(slots), side=sides[letter], action=action))
    return tuple(slots)


DRAFT_ORDER: tuple[DraftSlot, ...] = _build_order()


def slot_owner(slot: DraftSlot, our_side: Side) -> SlotOwner:
    """Return whether the slot is drafted by us or by the opponent."""
    return "us" if slot.side == our_side else "opponent"


def to_slot_model(slot: DraftSlot, our_side: Side) -> Slot:
    """Convert a draft slot to its response model from our point of view."""
    return Slot(
        index=slot.index,
        side=slot.side,
        action=slot.action,
        team=slot_owner(slot, our_side),
    )
