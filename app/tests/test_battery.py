from app.battery import diversify, _family
from app.catalog_loader import CatalogItem
from app.state_extraction import ConversationState

def _make_item(id: str, name: str) -> CatalogItem:
    return CatalogItem(
        id=id, name=name, keys=[], description="", url="",
        job_levels=[], duration_minutes=None, remote_supported=True,
        languages=[], search_blob=""
    )

def test_diversify_reduces_family_concentration():
    """Verify that diversify() limits the number of items per family to max_per_family."""
    # Create near-duplicate items (same family 'opq')
    item1 = _make_item("1", "OPQ32r Personality Questionnaire")
    item2 = _make_item("2", "OPQ Universal Competency Report")
    item3 = _make_item("3", "OPQ Leadership Report")
    item4 = _make_item("4", "OPQ MQ Sales Report")
    
    # An item from a different family
    item5 = _make_item("5", "Verify Interactive G+")

    # Mock scores (ordered strictly by score)
    scored = [
        (0.9, item1),
        (0.85, item2),
        (0.80, item3),
        (0.75, item4),
        (0.70, item5),
    ]

    state = ConversationState()
    # Apply diversify with max_per_family=2
    diversified = diversify(scored, state, max_per_family=2, max_per_category=3)

    assert len(diversified) == 3
    # The OPQ family should only have 2 members
    opq_count = sum(1 for _, item in diversified if _family(item) == "opq")
    assert opq_count == 2
    
    # The different family item should still be included
    verify_count = sum(1 for _, item in diversified if _family(item) == "verify")
    assert verify_count == 1
