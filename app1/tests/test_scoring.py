"""Unit tests for app.scoring – map_seniority_to_job_levels and score_candidate."""

import pytest

from app.scoring import map_seniority_to_job_levels, score_candidate
from app.catalog_loader import CatalogItem
from app.state_extraction import ConversationState


# ---------------------------------------------------------------------------
# Parametrised: (input phrase, expected labels)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "phrase, expected",
    [
        # Graduate / junior
        ("graduate", ["Graduate", "Entry-Level"]),
        ("fresh graduate", ["Graduate", "Entry-Level"]),
        ("trainee", ["Graduate", "Entry-Level"]),
        ("junior developer", ["Graduate", "Entry-Level"]),
        ("intern", ["Graduate", "Entry-Level"]),

        # Entry-level
        ("entry-level", ["Entry-Level"]),
        ("entry level associate", ["Entry-Level"]),

        # Mid-level / professional
        ("mid-level engineer", ["Mid-Professional", "Professional Individual Contributor"]),
        ("senior developer", ["Mid-Professional", "Professional Individual Contributor"]),
        ("experienced analyst", ["Mid-Professional", "Professional Individual Contributor"]),
        ("specialist", ["Mid-Professional", "Professional Individual Contributor"]),

        # Manager
        ("manager", ["Manager"]),
        ("senior manager", ["Manager", "Director"]),
        ("general manager", ["Manager", "Director"]),
        ("head of engineering", ["Manager", "Director"]),

        # Front-line / supervisor
        ("team lead", ["Front Line Manager", "Supervisor"]),
        ("team leader", ["Front Line Manager", "Supervisor"]),
        ("supervisor", ["Supervisor"]),
        ("frontline manager", ["Front Line Manager", "Supervisor"]),

        # Director
        ("director", ["Director"]),
        ("director of operations", ["Director"]),

        # C-suite / executive
        ("CXO", ["Executive", "Director"]),
        ("CEO", ["Executive", "Director"]),
        ("CFO", ["Executive", "Director"]),
        ("chief technology officer", ["Executive", "Director"]),
        ("c-suite executive", ["Executive", "Director"]),
        ("VP of Sales", ["Executive", "Director"]),

        # Fallback
        ("", ["General Population"]),
        ("something totally random", ["General Population"]),
    ],
)
def test_map_seniority(phrase: str, expected: list[str]) -> None:
    assert map_seniority_to_job_levels(phrase) == expected


# ---------------------------------------------------------------------------
# Case-insensitivity
# ---------------------------------------------------------------------------
def test_case_insensitive() -> None:
    """Mapping should be case-insensitive."""
    assert map_seniority_to_job_levels("GRADUATE") == ["Graduate", "Entry-Level"]
    assert map_seniority_to_job_levels("Manager") == ["Manager"]
    assert map_seniority_to_job_levels("cxo") == ["Executive", "Director"]


# ---------------------------------------------------------------------------
# score_candidate with new ConversationState booleans
# ---------------------------------------------------------------------------

def _make_item(**kwargs) -> CatalogItem:
    """Helper to create a CatalogItem with sensible defaults."""
    defaults = dict(
        id="test-1", name="Test Item", url="https://shl.com/test",
        job_levels=["Graduate", "Entry-Level"],
        duration_minutes=25, remote_supported=True,
        languages=["English (USA)"],
        keys=["Personality & Behavior"],
        description="A test assessment",
        search_blob="test assessment personality",
        embedding=None,
    )
    defaults.update(kwargs)
    return CatalogItem(**defaults)


def _make_state(**kwargs) -> ConversationState:
    """Helper to create a ConversationState with sensible defaults."""
    defaults = dict(
        role_title="Python Developer",
        domain_keywords=["python", "backend"],
        seniority_text="junior",
        language_required="English",
        wants_personality=True,
        wants_cognitive=False,
        wants_sjt=False,
        wants_simulation=False,
        wants_remote=True,
        duration_budget=30,
    )
    defaults.update(kwargs)
    return ConversationState(**defaults)


class TestScoreCandidate:
    """Tests for the composite scoring function."""

    def test_perfect_match_scores_high(self) -> None:
        """Item matching all constraints should score well."""
        item = _make_item()
        state = _make_state()
        score = score_candidate(item, state, cosine_sim=0.9)
        assert score > 0.8

    def test_bad_match_scores_low(self) -> None:
        """Item mismatching all constraints should score poorly."""
        item = _make_item(
            job_levels=["Director"],
            duration_minutes=90,
            remote_supported=False,
            languages=["French"],
            keys=["Simulations"],
        )
        state = _make_state()
        score = score_candidate(item, state, cosine_sim=0.1)
        assert score < 0.5

    def test_remote_flag_penalises_non_remote(self) -> None:
        """When wants_remote=True, non-remote items should score lower."""
        state = _make_state(wants_remote=True)
        remote_item = _make_item(remote_supported=True)
        non_remote_item = _make_item(remote_supported=False)
        s_remote = score_candidate(remote_item, state, 0.5)
        s_non = score_candidate(non_remote_item, state, 0.5)
        assert s_remote > s_non

    def test_remote_neutral_when_not_requested(self) -> None:
        """When wants_remote=False (default), both items score similarly."""
        state = _make_state(wants_remote=False)
        remote_item = _make_item(remote_supported=True)
        non_remote_item = _make_item(remote_supported=False)
        s_remote = score_candidate(remote_item, state, 0.5)
        s_non = score_candidate(non_remote_item, state, 0.5)
        # Should be equal (both get neutral 0.5)
        assert abs(s_remote - s_non) < 0.001

    def test_desired_keys_from_booleans(self) -> None:
        """The desired_keys property should derive from boolean flags."""
        state = _make_state(wants_personality=True, wants_cognitive=True)
        assert "Personality & Behavior" in state.desired_keys
        assert "Ability & Aptitude" in state.desired_keys
        assert "Simulations" not in state.desired_keys

    def test_duration_within_budget(self) -> None:
        """Items within duration budget should score higher."""
        state = _make_state(duration_budget=30)
        short = _make_item(duration_minutes=20)
        long = _make_item(duration_minutes=60)
        s_short = score_candidate(short, state, 0.5)
        s_long = score_candidate(long, state, 0.5)
        assert s_short > s_long

    def test_score_in_range(self) -> None:
        """Score should always be in [0, 1]."""
        state = _make_state()
        for sim in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            score = score_candidate(_make_item(), state, sim)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for sim={sim}"
