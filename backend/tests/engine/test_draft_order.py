from collections import Counter

from app.engine.draft_order import DRAFT_ORDER, slot_owner


def test_draft_order_has_twenty_slots_with_five_bans_and_picks_per_side() -> None:
    counts = Counter((slot.side, slot.action) for slot in DRAFT_ORDER)

    assert len(DRAFT_ORDER) == 20
    assert [slot.index for slot in DRAFT_ORDER] == list(range(20))
    assert counts == {
        ("blue", "ban"): 5,
        ("red", "ban"): 5,
        ("blue", "pick"): 5,
        ("red", "pick"): 5,
    }


def test_draft_order_follows_the_standard_pro_sequence() -> None:
    sequence = " ".join(f"{slot.side[0].upper()}{slot.action[0]}" for slot in DRAFT_ORDER)

    assert sequence == ("Bb Rb Bb Rb Bb Rb Bp Rp Rp Bp Bp Rp Rb Bb Rb Bb Rp Bp Bp Rp")


def test_slot_owner_maps_sides_to_us_or_opponent() -> None:
    first_slot = DRAFT_ORDER[0]
    second_slot = DRAFT_ORDER[1]

    assert slot_owner(first_slot, our_side="blue") == "us"
    assert slot_owner(first_slot, our_side="red") == "opponent"
    assert slot_owner(second_slot, our_side="red") == "us"
