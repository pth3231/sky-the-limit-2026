from typing import Any

from app.engine.draft_order import DRAFT_ORDER
from app.model.features import DraftState, FeatureSpace
from app.pipeline.dataset import GameDraft
from app.pipeline.ddragon import ChampionTraits
from app.pipeline.drafts import PlayerLine


def test_feature_space_fits_only_training_games_and_encodes_blue_orientation(
    make_game: Any,
) -> None:
    training_game = make_game(
        patch="16.1",
        blue_picks=["c1"],
        red_picks=["c2"],
        blue_bans=["c3"],
        red_bans=["c4"],
    ).model_copy(
        update={
            "lineups": [
                PlayerLine(
                    team_id="c9",
                    role="top",
                    champion_id="c1",
                    damage_share=0.2,
                    damage_taken=2000,
                    vision_score=30,
                    gold_share=0.25,
                    kill_participation=0.6,
                ),
                PlayerLine(team_id="opp", role="mid", champion_id="c2"),
            ]
        }
    )
    space = FeatureSpace.fit(
        [training_game],
        {"c1": ChampionTraits(tags=("Fighter",), info={"attack": 8})},
    )
    state = DraftState(
        patch="16.1",
        actions=["c3", "c4", "c3", "c4", "c3", "c4", "c1"],
    )
    encoded = space.encode(state)
    c1_index = space.champion_ids.index("c1")
    c3_index = space.champion_ids.index("c3")
    c4_index = space.champion_ids.index("c4")

    assert set(space.champion_ids) == {"c1", "c2", "c3", "c4"}
    assert "heldout-champion" not in space.champion_ids
    assert space.patches == ["16.1"]
    assert space.champion_roles["c1"] == "top"
    assert space.role_profiles["c1"]["top"].win_rate == 1
    assert space.role_profiles["c1"]["top"].damage_share == 0.2
    assert space.champion_traits["c1"].tags == ("Fighter",)
    # Check slot arrays for the encoded state
    assert c1_index in encoded.blue_picks
    assert c1_index not in encoded.red_picks
    assert c3_index in encoded.blue_bans
    assert c4_index in encoded.red_bans
    restored = FeatureSpace.model_validate_json(space.model_dump_json())
    assert restored.model_dump() == space.model_dump()


def test_feature_space_keeps_champions_with_unknown_roles_out_of_role_profiles(
    make_game: Any,
) -> None:
    game = make_game(blue_picks=["c1"]).model_copy(
        update={
            "lineups": [PlayerLine(team_id="c9", role=None, champion_id="c1")],
        }
    )

    space = FeatureSpace.fit([game])

    assert "c1" in space.champion_ids
    assert space.champion_roles == {}
    assert space.role_profiles == {}


def test_feature_space_keeps_unknown_patch_and_champion_dimensions_stable(
    make_game: Any,
) -> None:
    space = FeatureSpace.fit([make_game(blue_picks=["c1"])])

    known = space.encode(DraftState(patch="16.1", actions=[]))
    unknown = space.encode(DraftState(patch="unknown", actions=["unseen-champion"]))

    assert len(known.continuous) == len(unknown.continuous)
    # Verify that continuous features have stable dimensions regardless of patch/champion visibility
    action_count_size = len(DRAFT_ORDER) + 1
    # Patch features use len(patches) + 1 (the +1 is for unknown patch)
    patch_features_offset = action_count_size
    # For known patch "16.1", it should be in the known patches
    assert known.continuous[patch_features_offset] == 1.0
    # For unknown patch, it should be in the unknown patch slot (at the end)
    assert unknown.continuous[patch_features_offset + len(space.patches)] == 1.0


def test_feature_space_keeps_vocabulary_fixed_while_history_changes(make_game: Any) -> None:
    first_game = make_game(blue_picks=["c1"]).model_copy(
        update={
            "lineups": [
                PlayerLine(
                    team_id="c9",
                    role="mid",
                    champion_id="c1",
                    damage_share=0.2,
                )
            ],
        }
    )
    second_game = make_game(patch="16.2", winner="red", red_picks=["c2"]).model_copy(
        update={
            "lineups": [
                PlayerLine(
                    team_id="opp",
                    role="bot",
                    champion_id="c2",
                    damage_share=0.8,
                )
            ],
        }
    )
    vocabulary_games = [first_game, second_game]
    ddragon = {
        "c1": ChampionTraits(tags=("Fighter",), info={}),
        "c2": ChampionTraits(tags=("Mage",), info={}),
    }

    first_space = FeatureSpace.fit([first_game], ddragon, vocabulary_games=vocabulary_games)
    second_space = FeatureSpace.fit([second_game], ddragon, vocabulary_games=vocabulary_games)

    assert first_space.champion_ids == second_space.champion_ids == ["c1", "c2"]
    assert first_space.patches == second_space.patches == ["16.1", "16.2"]
    assert first_space.tag_names == second_space.tag_names == ["Fighter", "Mage"]
    assert first_space.continuous_size == second_space.continuous_size
    assert first_space.champion_roles == {"c1": "mid"}
    assert second_space.champion_roles == {"c2": "bot"}
    assert first_space.role_profiles["c1"]["mid"].damage_share == 0.2
    assert second_space.role_profiles["c2"]["bot"].damage_share == 0.8


def test_encode_returns_sentinel_padded_slot_arrays_for_empty_state(
    synthetic_drafts: list[GameDraft],
) -> None:
    feature_space = FeatureSpace.fit(synthetic_drafts)
    sentinel = feature_space.num_champions

    encoded = feature_space.encode(DraftState())

    assert encoded.blue_picks == [sentinel] * 5
    assert encoded.blue_bans == [sentinel] * 5
    assert encoded.red_picks == [sentinel] * 5
    assert encoded.red_bans == [sentinel] * 5


def test_encode_places_champion_indices_in_the_matching_slot_group(
    synthetic_drafts: list[GameDraft],
) -> None:
    feature_space = FeatureSpace.fit(synthetic_drafts)
    full_game = synthetic_drafts[0]

    # Reorder game actions to match DRAFT_ORDER sequence
    # Build slot assignments for each action based on DRAFT_ORDER
    draft_actions: list[str] = [""] * len(DRAFT_ORDER)

    # Count picks/bans per side to map to their DRAFT_ORDER positions
    action_counters = {"blue_bans": 0, "red_bans": 0, "blue_picks": 0, "red_picks": 0}

    for action in full_game.actions:
        side = "blue" if action.team_id == full_game.blue_team_id else "red"
        action_type = "bans" if action.action_type == "ban" else "picks"
        counter_key = f"{side}_{action_type}"

        # Find the next DRAFT_ORDER slot for this type/side combination
        slot_index = 0
        count_needed = action_counters[counter_key]
        count_found = 0
        for slot in DRAFT_ORDER:
            slot_key = f"{slot.side}_{'bans' if slot.action == 'ban' else 'picks'}"
            if slot_key == counter_key:
                if count_found == count_needed:
                    draft_actions[slot_index] = action.champion_id
                    break
                count_found += 1
            slot_index += 1

        action_counters[counter_key] += 1

    state = DraftState(
        patch=full_game.patch,
        actions=draft_actions,
    )

    encoded = feature_space.encode(state)

    champion_index = {
        champion_id: index for index, champion_id in enumerate(feature_space.champion_ids)
    }
    expected_blue_picks = sorted(
        champion_index[action.champion_id]
        for action in full_game.actions
        if action.team_id == full_game.blue_team_id and action.action_type == "pick"
    )
    encoded_blue_picks = sorted(
        value for value in encoded.blue_picks if value != feature_space.num_champions
    )
    assert encoded_blue_picks == expected_blue_picks
    assert len(encoded.blue_picks) == 5


def test_champion_index_is_cached_across_encode_calls(make_game: Any) -> None:
    games = [make_game(blue_picks=["c0", "c1"], red_picks=["c2", "c3"])]
    feature_space = FeatureSpace.fit(games)

    first = feature_space._champion_index
    second = feature_space._champion_index

    assert first is second
