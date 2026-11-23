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
    # Check the draft sequence at its known positions (actions=["c3","c4","c3","c4","c3","c4","c1"];
    # DRAFT_ORDER[0..5] are bans B,R,B,R,B,R, DRAFT_ORDER[6] is the first blue pick)
    assert encoded.draft_sequence[0] == c3_index
    assert encoded.draft_sequence[1] == c4_index
    assert encoded.draft_sequence[6] == c1_index
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


def test_encode_returns_sentinel_padded_sequence_for_empty_state(
    synthetic_drafts: list[GameDraft],
) -> None:
    feature_space = FeatureSpace.fit(synthetic_drafts)
    sentinel = feature_space.num_champions

    encoded = feature_space.encode(DraftState())

    assert encoded.draft_sequence == [sentinel] * len(DRAFT_ORDER)


def test_encode_places_champion_indices_at_their_draft_order_position(
    synthetic_drafts: list[GameDraft],
) -> None:
    feature_space = FeatureSpace.fit(synthetic_drafts)
    full_game = synthetic_drafts[0]
    action_ids = [action.champion_id for action in full_game.actions[: len(DRAFT_ORDER)]]
    state = DraftState(patch=full_game.patch, actions=action_ids)

    encoded = feature_space.encode(state)

    champion_index = {
        champion_id: index for index, champion_id in enumerate(feature_space.champion_ids)
    }
    expected = [
        champion_index.get(champion_id, feature_space.num_champions) for champion_id in action_ids
    ]
    expected += [feature_space.num_champions] * (len(DRAFT_ORDER) - len(expected))
    assert encoded.draft_sequence == expected


def test_champion_index_is_cached_across_encode_calls(make_game: Any) -> None:
    games = [make_game(blue_picks=["c0", "c1"], red_picks=["c2", "c3"])]
    feature_space = FeatureSpace.fit(games)

    first = feature_space._champion_index
    second = feature_space._champion_index

    assert first is second


def test_feature_space_encodes_matchup_win_rate_between_same_role_picks(make_game: Any) -> None:
    game = make_game(blue_picks=["c1"], red_picks=["c2"]).model_copy(
        update={
            "lineups": [
                PlayerLine(team_id="c9", role="top", champion_id="c1"),
                PlayerLine(team_id="opp", role="top", champion_id="c2"),
            ],
        }
    )
    space = FeatureSpace.fit([game])
    # The first six DRAFT_ORDER slots are bans (blue, red, blue, red, blue, red); the
    # seventh and eighth are the first blue and red picks. Pad with filler bans so
    # c1/c2 land in actual pick slots rather than the leading ban slots.
    state = DraftState(patch="16.1", actions=["b1", "b2", "b3", "b4", "b5", "b6", "c1", "c2"])
    encoded = space.encode(state)

    per_side_width = 5 + 7 + len(space.tag_names) + 4 + 6 + 1 + 1
    blue_start = len(DRAFT_ORDER) + 1 + len(space.patches) + 1
    blue_matchup_index = blue_start + per_side_width - 2
    red_matchup_index = blue_start + 2 * per_side_width - 2

    # c1 beat c2 in their one recorded top matchup
    assert encoded.continuous[blue_matchup_index] == 1.0
    assert encoded.continuous[red_matchup_index] == 0.0  # c2 lost that matchup

    lone_pick_encoded = space.encode(
        DraftState(patch="16.1", actions=["b1", "b2", "b3", "b4", "b5", "b6", "c1"])
    )
    assert lone_pick_encoded.continuous[blue_matchup_index] == 0.0  # no opposing top pick yet


def test_feature_space_encodes_team_win_rate_as_a_strength_prior(make_game: Any) -> None:
    strong_games = [make_game(blue="strong", red="filler", winner="blue") for _ in range(12)]
    weak_games = [make_game(blue="weak", red="filler", winner="red") for _ in range(12)]
    space = FeatureSpace.fit([*strong_games, *weak_games])

    per_side_width = 5 + 7 + len(space.tag_names) + 4 + 6 + 1 + 1
    blue_start = len(DRAFT_ORDER) + 1 + len(space.patches) + 1
    blue_team_index = blue_start + per_side_width - 1
    red_team_index = blue_start + 2 * per_side_width - 1

    strong_encoded = space.encode(
        DraftState(patch="16.1", blue_team_id="strong", red_team_id="filler")
    )
    weak_encoded = space.encode(DraftState(patch="16.1", blue_team_id="weak", red_team_id="filler"))
    unknown_encoded = space.encode(
        DraftState(patch="16.1", blue_team_id="never-seen", red_team_id="filler")
    )

    assert red_team_index == len(strong_encoded.continuous) - 2
    assert strong_encoded.continuous[blue_team_index] > 0.5
    assert weak_encoded.continuous[blue_team_index] < 0.5
    assert unknown_encoded.continuous[blue_team_index] == 0.5


def test_team_win_rate_reflects_recent_form_over_stale_history(make_game: Any) -> None:
    # More old losses than the recency window, so an all-time average would stay well
    # below 0.5; only a recency-limited rate can land above it.
    old_losses = [
        make_game(blue="roster-changed", red="filler", winner="red").model_copy(
            update={"started_at": f"2024-{month:02d}-{day:02d}T12:00:00Z"}
        )
        for month in range(1, 9)
        for day in range(1, 14)
    ]
    recent_wins = [
        make_game(blue="roster-changed", red="filler", winner="blue").model_copy(
            update={"started_at": f"2026-01-{day:02d}T12:00:00Z"}
        )
        for day in range(1, 31)
    ]
    space = FeatureSpace.fit([*old_losses, *recent_wins])

    rate = space.team_win_rates["roster-changed"]

    # Dominated by 5 recent wins, not diluted by 24 old (pre-roster-change) losses.
    assert rate > 0.5


def test_feature_space_fits_aggregates_from_training_games_only(make_game: Any) -> None:
    from app.pipeline.aggregates import compute_aggregates

    history_games = [make_game(blue_picks=["c0"], red_picks=["c1"]) for _ in range(3)]
    future_games = [make_game(patch="16.2", blue_picks=["c2"], red_picks=["c3"]) for _ in range(3)]

    feature_space = FeatureSpace.fit(
        history_games, vocabulary_games=[*history_games, *future_games]
    )

    assert feature_space.aggregates == compute_aggregates(history_games)


def test_encode_appends_a_scorer_probability_as_the_last_continuous_feature(
    make_game: Any,
) -> None:
    # "strong" wins almost every game it's picked in; "weak" wins almost none.
    strong_wins = [
        make_game(blue_picks=["strong"], red_picks=["weak"], winner="blue") for _ in range(20)
    ]
    weak_wins = [
        make_game(blue_picks=["weak"], red_picks=["strong"], winner="blue") for _ in range(2)
    ]
    feature_space = FeatureSpace.fit([*strong_wins, *weak_wins])

    blue_pick_index = next(
        index
        for index, slot in enumerate(DRAFT_ORDER)
        if slot.side == "blue" and slot.action == "pick"
    )
    red_pick_index = next(
        index
        for index, slot in enumerate(DRAFT_ORDER)
        if slot.side == "red" and slot.action == "pick"
    )
    actions = ["filler"] * len(DRAFT_ORDER)
    actions[blue_pick_index] = "strong"
    actions[red_pick_index] = "weak"
    state = DraftState(patch="16.1", actions=actions, blue_team_id="c9", red_team_id="opp")

    encoded = feature_space.encode(state)

    assert 0.0 <= encoded.continuous[-1] <= 1.0
    assert encoded.continuous[-1] > 0.5


def test_encode_scorer_probability_defaults_to_neutral_without_training_data() -> None:
    feature_space = FeatureSpace.fit([])

    encoded = feature_space.encode(DraftState())

    assert encoded.continuous[-1] == 0.5


def test_encode_caches_blended_stats_by_patch_not_by_team_pair(
    make_game: Any, monkeypatch: Any
) -> None:
    from app.model import features as features_module

    games = [make_game(patch="16.1", blue_picks=["c0"], red_picks=["c1"]) for _ in range(5)]
    feature_space = FeatureSpace.fit(games)

    call_count = 0
    original_blend_patches = features_module.blend_patches

    def counting_blend_patches(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original_blend_patches(*args, **kwargs)

    monkeypatch.setattr(features_module, "blend_patches", counting_blend_patches)

    for team_index in range(10):
        feature_space.encode(
            DraftState(
                patch="16.1",
                actions=["c0", "c1"],
                blue_team_id=f"team-{team_index}",
                red_team_id=f"opp-{team_index}",
            )
        )

    assert call_count <= 1
