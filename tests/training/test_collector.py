"""Tests for training.collector — training data capture module."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(enabled=True, exclude=None, base_path=None):
    td = {
        "enabled": enabled,
        "exclude_session_types": exclude or [],
        "capture_system_prompts": True,
        "capture_auto_injections": True,
    }
    if base_path:
        td["base_path"] = str(base_path)
    return {"training_data": td}


def _make_db(session_row=None, messages=None):
    db = MagicMock()
    db.get_session.return_value = session_row or {
        "id": "test_session_id",
        "source": "cli",
        "started_at": 1748693000.0,
        "ended_at": 1748693100.0,
        "end_reason": "user_exit",
        "system_prompt": "You are Sou.",
        "model": "claude-sonnet-4-6",
        "input_tokens": 100,
        "output_tokens": 50,
    }
    db.get_messages.return_value = messages or []
    return db


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------

class TestIsEnabled:
    def test_disabled_by_default(self):
        from training.collector import is_enabled
        with patch("training.collector._load_td_config", return_value=None):
            assert is_enabled() is False

    def test_enabled_when_config_says_true(self):
        from training.collector import is_enabled
        cfg = _make_config(enabled=True)
        assert is_enabled(cfg) is True

    def test_disabled_when_config_says_false(self):
        from training.collector import is_enabled
        cfg = _make_config(enabled=False)
        assert is_enabled(cfg) is False

    def test_enabled_with_nested_config(self):
        from training.collector import is_enabled
        cfg = _make_config(enabled=True)
        # Passing the full config dict (not just training_data sub-dict)
        assert is_enabled(cfg) is True


# ---------------------------------------------------------------------------
# _make_folder_name
# ---------------------------------------------------------------------------

class TestMakeFolderName:
    def test_format(self):
        from training.collector import _make_folder_name
        ts = datetime(2026, 5, 31, 14, 8, 43, tzinfo=timezone.utc)
        name = _make_folder_name(ts, "autonomy", "autonomy_20260531_140843")
        assert name.startswith("2026-05-31_14-08-43_autonomy_")

    def test_short_id_from_session_id(self):
        from training.collector import _make_folder_name
        ts = datetime(2026, 5, 31, 14, 8, 43, tzinfo=timezone.utc)
        name = _make_folder_name(ts, "cli", "abcdef123456")
        assert name.endswith("123456")

    def test_short_session_id(self):
        from training.collector import _make_folder_name
        ts = datetime(2026, 5, 31, 14, 8, 43, tzinfo=timezone.utc)
        name = _make_folder_name(ts, "cli", "abc")
        assert "abc" in name

    def test_special_chars_in_type_sanitized(self):
        from training.collector import _make_folder_name
        ts = datetime(2026, 5, 31, 14, 8, 43, tzinfo=timezone.utc)
        name = _make_folder_name(ts, "tele-gram!", "sess123456")
        assert "/" not in name
        assert "!" not in name


# ---------------------------------------------------------------------------
# _utc_iso
# ---------------------------------------------------------------------------

class TestUtcIso:
    def test_none_returns_empty(self):
        from training.collector import _utc_iso
        assert _utc_iso(None) == ""

    def test_zero_returns_empty(self):
        from training.collector import _utc_iso
        assert _utc_iso(0) == ""

    def test_unix_timestamp(self):
        from training.collector import _utc_iso
        # 1798761600.0 = 2027-01-01T00:00:00Z
        ts = 1798761600.0
        result = _utc_iso(ts)
        assert result.endswith("Z")
        assert "T" in result
        assert "2027" in result

    def test_format_matches_iso8601(self):
        from training.collector import _utc_iso
        result = _utc_iso(1748693000.0)
        # Should be parseable as ISO 8601
        parsed = datetime.strptime(result, "%Y-%m-%dT%H:%M:%SZ")
        assert parsed is not None


# ---------------------------------------------------------------------------
# _write_json_safe
# ---------------------------------------------------------------------------

class TestWriteJsonSafe:
    def test_writes_valid_json(self, tmp_path):
        from training.collector import _write_json_safe
        path = tmp_path / "out.json"
        data = {"key": "value", "num": 42}
        result = _write_json_safe(path, data)
        assert result is True
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_returns_false_on_unserializable(self, tmp_path):
        from training.collector import _write_json_safe
        path = tmp_path / "bad.json"

        class Unserializable:
            pass

        # default=str should handle most cases; use a circular ref to force failure
        data: dict = {}
        data["self"] = data  # circular reference — json.dumps will raise
        result = _write_json_safe(path, data)
        assert result is False
        assert not path.exists()

    def test_atomic_write(self, tmp_path):
        from training.collector import _write_json_safe
        path = tmp_path / "atomic.json"
        _write_json_safe(path, {"a": 1})
        assert path.exists()
        # No leftover .tmp file
        assert not (tmp_path / "atomic.tmp").exists()


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_empty_session(self):
        from training.collector import _build_messages
        db = MagicMock()
        db.get_messages.return_value = []
        assert _build_messages(db, "sid") == []

    def test_user_turn_increments_turn(self):
        from training.collector import _build_messages
        db = MagicMock()
        db.get_messages.return_value = [
            {"role": "user", "content": "hello", "tool_calls": None, "tool_call_id": None, "timestamp": 1748693000.0},
            {"role": "assistant", "content": "hi", "tool_calls": None, "tool_call_id": None, "timestamp": 1748693001.0},
            {"role": "user", "content": "bye", "tool_calls": None, "tool_call_id": None, "timestamp": 1748693002.0},
        ]
        messages = _build_messages(db, "sid")
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert user_msgs[0]["turn"] == 1
        assert user_msgs[1]["turn"] == 2

    def test_tool_calls_linked_to_results(self):
        from training.collector import _build_messages
        db = MagicMock()
        tool_calls = json.dumps([{
            "id": "call_abc",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
        }])
        db.get_messages.return_value = [
            {"role": "user", "content": "read it", "tool_calls": None, "tool_call_id": None, "timestamp": 1748693000.0},
            {"role": "assistant", "content": None, "tool_calls": tool_calls, "tool_call_id": None, "timestamp": 1748693001.0},
            {"role": "tool", "content": "file contents here", "tool_calls": None, "tool_call_id": "call_abc", "timestamp": 1748693002.0},
        ]
        messages = _build_messages(db, "sid")
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        tc = assistant_msgs[0]["tool_calls"][0]
        assert tc["tool_name"] == "read_file"
        assert tc["arguments"] == {"path": "/tmp/x"}
        assert tc["result"] == "file contents here"
        assert tc["success"] is True
        assert tc["error"] is None

    def test_tool_messages_not_in_output(self):
        from training.collector import _build_messages
        db = MagicMock()
        tool_calls = json.dumps([{
            "id": "call_xyz",
            "function": {"name": "some_tool", "arguments": "{}"},
        }])
        db.get_messages.return_value = [
            {"role": "user", "content": "go", "tool_calls": None, "tool_call_id": None, "timestamp": 1748693000.0},
            {"role": "assistant", "content": "", "tool_calls": tool_calls, "tool_call_id": None, "timestamp": 1748693001.0},
            {"role": "tool", "content": "ok", "tool_calls": None, "tool_call_id": "call_xyz", "timestamp": 1748693002.0},
        ]
        messages = _build_messages(db, "sid")
        roles = [m["role"] for m in messages]
        assert "tool" not in roles

    def test_error_result_detected(self):
        from training.collector import _build_messages
        db = MagicMock()
        tool_calls = json.dumps([{"id": "c1", "function": {"name": "boom", "arguments": "{}"}}])
        db.get_messages.return_value = [
            {"role": "user", "content": "x", "tool_calls": None, "tool_call_id": None, "timestamp": 1748693000.0},
            {"role": "assistant", "content": None, "tool_calls": tool_calls, "tool_call_id": None, "timestamp": 1748693001.0},
            {"role": "tool", "content": "Error: something went wrong", "tool_calls": None, "tool_call_id": "c1", "timestamp": 1748693002.0},
        ]
        messages = _build_messages(db, "sid")
        tc = messages[1]["tool_calls"][0]
        assert tc["success"] is False
        assert tc["error"] == "Error: something went wrong"

    def test_missing_tool_result(self):
        from training.collector import _build_messages
        db = MagicMock()
        tool_calls = json.dumps([{"id": "orphan", "function": {"name": "x", "arguments": "{}"}}])
        db.get_messages.return_value = [
            {"role": "user", "content": "x", "tool_calls": None, "tool_call_id": None, "timestamp": 1748693000.0},
            {"role": "assistant", "content": None, "tool_calls": tool_calls, "tool_call_id": None, "timestamp": 1748693001.0},
            # No tool result row for "orphan"
        ]
        messages = _build_messages(db, "sid")
        tc = messages[1]["tool_calls"][0]
        assert tc["success"] is False
        assert tc["error"] == "no result recorded"

    def test_get_messages_failure_returns_empty(self):
        from training.collector import _build_messages
        db = MagicMock()
        db.get_messages.side_effect = RuntimeError("DB error")
        assert _build_messages(db, "sid") == []


# ---------------------------------------------------------------------------
# capture_wake
# ---------------------------------------------------------------------------

class TestCaptureWake:
    def test_disabled_is_noop(self, tmp_path):
        from training.collector import capture_wake
        cfg = _make_config(enabled=False, base_path=tmp_path)
        capture_wake({"wake_session_id": "w1", "timestamp": "2026-05-31T00:00:00Z"}, config=cfg)
        assert not (tmp_path / "wake").exists()

    def test_creates_wake_json(self, tmp_path):
        from training.collector import capture_wake
        cfg = _make_config(base_path=tmp_path)
        wake_data = {
            "wake_session_id": "wake_20260531_140843",
            "timestamp": "2026-05-31T14:08:43Z",
            "schema_version": "1.0",
            "direction": "outward",
            "decision": "wake",
            "resulted_in_session_id": "autonomy_20260531_140900",
        }
        capture_wake(wake_data, config=cfg)
        folders = list((tmp_path / "wake").iterdir())
        assert len(folders) == 1
        wake_json = folders[0] / "wake.json"
        assert wake_json.exists()
        data = json.loads(wake_json.read_text())
        assert data["schema_version"] == "1.0"
        assert data["decision"] == "wake"

    def test_discards_circular_reference(self, tmp_path):
        from training.collector import capture_wake
        cfg = _make_config(base_path=tmp_path)
        bad: dict = {}
        bad["self"] = bad  # circular — json.dumps will raise
        capture_wake(bad, config=cfg)
        # No file should be written
        wake_dir = tmp_path / "wake"
        assert not wake_dir.exists() or len(list(wake_dir.iterdir())) == 0

    def test_excluded_type_is_noop(self, tmp_path):
        from training.collector import capture_wake
        cfg = _make_config(base_path=tmp_path, exclude=["wake"])
        capture_wake({"wake_session_id": "w1"}, config=cfg)
        assert not (tmp_path / "wake").exists()


# ---------------------------------------------------------------------------
# capture_session
# ---------------------------------------------------------------------------

class TestCaptureSession:
    def test_disabled_is_noop(self, tmp_path):
        from training.collector import capture_session
        cfg = _make_config(enabled=False, base_path=tmp_path)
        db = _make_db()
        capture_session("sid_20260531_140843_abcdef12", "cli", db, config=cfg)
        assert not (tmp_path / "sessions").exists()

    def test_excluded_session_type_is_noop(self, tmp_path):
        from training.collector import capture_session
        cfg = _make_config(base_path=tmp_path, exclude=["cli"])
        db = _make_db()
        capture_session("sid_20260531_140843_abcdef12", "cli", db, config=cfg)
        assert not (tmp_path / "sessions").exists()

    def test_creates_session_json(self, tmp_path):
        from training.collector import capture_session
        # Clear registry so tmp_path is used
        import training.collector as _mod
        _mod._folder_registry.clear()

        cfg = _make_config(base_path=tmp_path)
        db = _make_db()
        capture_session("20260531_140843_abcdef12", "cli", db, config=cfg)

        folders = list((tmp_path / "sessions").iterdir())
        assert len(folders) == 1
        session_json = folders[0] / "session.json"
        assert session_json.exists()

        data = json.loads(session_json.read_text())
        assert data["schema_version"] == "1.0"
        assert data["session_id"] == "20260531_140843_abcdef12"
        assert data["session_type"] == "cli"
        assert "metadata" in data
        assert "messages" in data

    def test_system_prompt_omitted_when_disabled(self, tmp_path):
        from training.collector import capture_session
        import training.collector as _mod
        _mod._folder_registry.clear()

        cfg = _make_config(base_path=tmp_path)
        cfg["training_data"]["capture_system_prompts"] = False
        db = _make_db()
        capture_session("20260531_140900_bbbbbb12", "cli", db, config=cfg)

        folders = list((tmp_path / "sessions").iterdir())
        data = json.loads((folders[0] / "session.json").read_text())
        assert data["system_prompt"] == ""


# ---------------------------------------------------------------------------
# capture_autonomy
# ---------------------------------------------------------------------------

class TestCaptureAutonomy:
    def test_disabled_is_noop(self, tmp_path):
        from training.collector import capture_autonomy
        cfg = _make_config(enabled=False, base_path=tmp_path)
        db = _make_db()
        capture_autonomy("autonomy_20260531_140843", db, {}, config=cfg)
        assert not (tmp_path / "autonomy").exists()

    def test_creates_session_json_with_autonomy_metadata(self, tmp_path):
        from training.collector import capture_autonomy
        import training.collector as _mod
        _mod._folder_registry.clear()

        cfg = _make_config(base_path=tmp_path)
        db = _make_db()
        decision = {
            "wake": True,
            "session_type": "outward",
            "reason": "user pinged",
            "signals": [
                {"signal_id": "discord:123/msg:456", "action": "respond"},
            ],
        }
        capture_autonomy(
            "autonomy_20260531_140843", db, decision,
            extra_meta={"wake_session_id": "wake_20260531_140800"},
            config=cfg,
        )

        folders = list((tmp_path / "autonomy").iterdir())
        assert len(folders) == 1
        data = json.loads((folders[0] / "session.json").read_text())
        assert "autonomy_metadata" in data["metadata"]
        meta = data["metadata"]["autonomy_metadata"]
        assert meta["trigger_source"] == "discord"
        assert meta["wake_session_id"] == "wake_20260531_140800"

    def test_trigger_source_classification(self):
        from training.collector import _classify_trigger_source
        assert _classify_trigger_source([{"signal_id": "discord:c/msg:m"}]) == "discord"
        assert _classify_trigger_source([{"signal_id": "contact:alice"}]) == "contact"
        assert _classify_trigger_source([{"signal_id": "intent:1"}]) == "intent"
        assert _classify_trigger_source([{"signal_id": "curiosity:abc"}]) == "curiosity"
        assert _classify_trigger_source([]) == "autonomous"
        assert _classify_trigger_source([
            {"signal_id": "discord:x"},
            {"signal_id": "contact:y"},
        ]) == "mixed"


# ---------------------------------------------------------------------------
# _wake_decision_str
# ---------------------------------------------------------------------------

class TestWakeDecisionStr:
    def test_fired_true_returns_wake(self):
        from training.collector import _wake_decision_str
        assert _wake_decision_str({"wake": True}, fired=True, suppressed=False) == "wake"

    def test_wake_false_returns_skip(self):
        from training.collector import _wake_decision_str
        assert _wake_decision_str({"wake": False, "signals": []}, fired=False, suppressed=False) == "skip"

    def test_wake_false_with_hold_signals(self):
        from training.collector import _wake_decision_str
        decision = {"wake": False, "signals": [{"action": "hold"}]}
        assert _wake_decision_str(decision, fired=False, suppressed=False) == "hold"

    def test_wake_false_with_self_reflect(self):
        from training.collector import _wake_decision_str
        decision = {"wake": False, "signals": [{"action": "self_reflect"}]}
        assert _wake_decision_str(decision, fired=False, suppressed=False) == "journal"

    def test_suppressed_returns_hold(self):
        from training.collector import _wake_decision_str
        decision = {"wake": True, "signals": []}
        assert _wake_decision_str(decision, fired=False, suppressed=True) == "hold"


# ---------------------------------------------------------------------------
# capture_attachment
# ---------------------------------------------------------------------------

class TestCaptureAttachment:
    def test_disabled_returns_none(self, tmp_path):
        from training.collector import capture_attachment
        cfg = _make_config(enabled=False, base_path=tmp_path)
        result = capture_attachment("sid", "telegram", "/nonexistent/file.jpg", "file.jpg", "image/jpeg", config=cfg)
        assert result is None

    def test_copies_file_to_attachments_dir(self, tmp_path):
        from training.collector import capture_attachment
        import training.collector as _mod
        _mod._folder_registry.clear()

        cfg = _make_config(base_path=tmp_path)
        src = tmp_path / "test_img.jpg"
        src.write_bytes(b"fake image data")

        result = capture_attachment(
            "20260531_140843_abcdef12", "telegram",
            str(src), "test_img.jpg", "image/jpeg",
            config=cfg,
        )
        assert result == "test_img.jpg"

        folder = _mod._folder_registry.get("20260531_140843_abcdef12")
        assert folder is not None
        dest = folder / "attachments" / "test_img.jpg"
        assert dest.exists()
        assert dest.read_bytes() == b"fake image data"

    def test_missing_file_returns_none(self, tmp_path):
        from training.collector import capture_attachment
        import training.collector as _mod
        _mod._folder_registry.clear()

        cfg = _make_config(base_path=tmp_path)
        result = capture_attachment(
            "20260531_141000_xxxxxxxx", "telegram",
            "/nonexistent/path.jpg", "photo.jpg", "image/jpeg",
            config=cfg,
        )
        assert result is None

    def test_creates_attachments_json(self, tmp_path):
        from training.collector import capture_attachment
        import training.collector as _mod
        _mod._folder_registry.clear()

        cfg = _make_config(base_path=tmp_path)
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"pdf bytes")

        capture_attachment(
            "20260531_141100_yyyyyyyy", "telegram",
            str(src), "doc.pdf", "application/pdf",
            config=cfg,
        )

        folder = _mod._folder_registry["20260531_141100_yyyyyyyy"]
        meta = json.loads((folder / "attachments" / "attachments.json").read_text())
        assert len(meta) == 1
        assert meta[0]["filename"] == "doc.pdf"
        assert meta[0]["media_type"] == "application/pdf"


# ---------------------------------------------------------------------------
# capture_cron
# ---------------------------------------------------------------------------

class TestCaptureCron:
    def test_disabled_is_noop(self, tmp_path):
        from training.collector import capture_cron
        cfg = _make_config(enabled=False, base_path=tmp_path)
        db = _make_db()
        capture_cron("cron_abc_20260531_140843", {"name": "test"}, db, config=cfg)
        assert not (tmp_path / "cron").exists()

    def test_creates_cron_json(self, tmp_path):
        from training.collector import capture_cron
        import training.collector as _mod
        _mod._folder_registry.clear()

        cfg = _make_config(base_path=tmp_path)
        db = _make_db()
        job = {"name": "daily-digest", "schedule_display": "0 9 * * *"}
        capture_cron("cron_abc_20260531_140843", job, db, config=cfg)

        folders = list((tmp_path / "cron").iterdir())
        assert len(folders) == 1
        cron_json = folders[0] / "cron.json"
        assert cron_json.exists()

        data = json.loads(cron_json.read_text())
        assert data["schema_version"] == "1.0"
        assert data["cron_job_name"] == "daily-digest"
        assert data["metadata"]["cron_schedule"] == "0 9 * * *"
