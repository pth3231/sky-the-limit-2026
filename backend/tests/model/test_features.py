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

    # Construct a state with specific champions at known DRAFT_ORDER positions
    # DRAFT_ORDER positions:
    # 0: ban, blue
    # 1: ban, red
    # 2-5: more bans
    # 6: pick, blue -> c0
    # 7: pick, red -> c2
    # 8: pick, red -> c3
    # 9: pick, blue -> c1
    # 10+: remaining picks and bans

    # Build state.actions with sentinels (we use a placeholder string) for missing slots,
    # then fill in the ones we care about
    state = DraftState(
        patch="16.1",
        actions=["", "", "", "", "", "", "c0", "c2", "c3", "c1"],
    )

    encoded = feature_space.encode(state)

    champion_index = {
        champion_id: index for index, champion_id in enumerate(feature_space.champion_ids)
    }

    # Verify that the picks end up in the right slot groups
    assert champion_index["c0"] in encoded.blue_picks
    assert champion_index["c1"] in encoded.blue_picks
    assert champion_index["c2"] in encoded.red_picks
    assert champion_index["c3"] in encoded.red_picks

    # Verify slot array lengths
    assert len(encoded.blue_picks) == 5
    assert len(encoded.red_picks) == 5
