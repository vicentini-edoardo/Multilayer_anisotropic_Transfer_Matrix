from __future__ import annotations

from multilayer_atm.ui import material_builder as builder


def test_dismissed_custom_material_dialog_clears_open_state_and_draft(monkeypatch):
    state = {
        builder.CUSTOM_MATERIAL_DIALOG_OPEN_KEY: True,
        builder.CUSTOM_MATERIAL_DIALOG_EDIT_MODE_KEY: True,
        builder.CUSTOM_MATERIAL_DIALOG_ORIGINAL_NAME_KEY: "old",
        "custom_material_dialog_name": "unsaved",
    }
    monkeypatch.setattr(builder.st, "session_state", state)

    builder._dismiss_custom_material_dialog()

    assert state[builder.CUSTOM_MATERIAL_DIALOG_OPEN_KEY] is False
    assert state[builder.CUSTOM_MATERIAL_DIALOG_EDIT_MODE_KEY] is False
    assert state[builder.CUSTOM_MATERIAL_DIALOG_ORIGINAL_NAME_KEY] is None
    assert "custom_material_dialog_name" not in state


def test_new_custom_material_starts_without_abandoned_draft(monkeypatch):
    state = {
        "mat_1": builder.ADD_NEW_MATERIAL_OPTION,
        "custom_material_dialog_name": "abandoned",
    }
    monkeypatch.setattr(builder.st, "session_state", state)

    builder.handle_material_selection_change("mat_1", "SiC3C")

    assert "custom_material_dialog_name" not in state
    assert state[builder.CUSTOM_MATERIAL_DIALOG_OPEN_KEY] is True
    assert state["mat_1"] == "SiC3C"
