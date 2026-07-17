"""Behavioral tests for the pure ``Session`` state machine.

`Session` takes no clock and no threads (DESIGN D11) — every test below
drives it with hand-picked, strictly-increasing `t` floats and injected
`present_ids` sets. No real sleeps, no wall clock, anywhere in this suite.

Two debounce constants recur throughout (the frozen constructor defaults,
also spelled out by the ACs themselves): `off_debounce_s=1.5`,
`on_debounce_s=1.0`. Tests bracket transitions well clear of those
thresholds (>=1.5x margin) rather than probing the exact boundary, since the
ACs care about *behavior* (a debounced flip happens / doesn't happen), not
the precise tunable value.

T10 (B-S) widens `observe` to `observe(t, present_ids, roster=None)`. The
default keeps every test above it semantically untouched — an omitted roster
still means "filter nothing" — so the AC1-AC11 suites are unmigrated by design,
and `TestBS2` below pins that default down as a promise rather than an
accident.
"""

from __future__ import annotations

import pytest

from backend.session import (
    InstrumentReport,
    InstrumentStatus,
    InvalidPhase,
    Phase,
    Report,
    Session,
    SetupReadiness,
    UsageWindow,
)


def _confirm_presence(
    session: Session, ids: frozenset[int], first_t: float, second_t: float
) -> None:
    """Feed two observes (same `ids`) at `first_t` then `second_t`, with the
    gap well past the default entry debounce (1.0s), so every id in `ids` is
    a confirmed member of the universe before the test's real scenario
    starts. Used only after `start()`, inside RECORDING."""
    session.observe(first_t, ids)
    session.observe(second_t, ids)


class TestAC1PhaseStateMachine:
    """AC1: fresh session is SETUP; start/stop/start walks
    SETUP->RECORDING->FINISHED->RECORDING; every invalid transition and
    wrong-phase accessor raises InvalidPhase."""

    def test_ac1_fresh_session_starts_in_setup(self) -> None:
        assert Session().phase == Phase.SETUP

    def test_ac1_full_lifecycle_walks_through_every_phase(self) -> None:
        session = Session()

        session.start(0.0)
        assert session.phase == Phase.RECORDING

        report = session.stop(5.0)
        assert session.phase == Phase.FINISHED
        assert isinstance(report, Report)

        session.start(6.0)
        assert session.phase == Phase.RECORDING

    def test_ac1_start_while_recording_raises_invalid_phase(self) -> None:
        session = Session()
        session.start(0.0)

        with pytest.raises(InvalidPhase):
            session.start(1.0)

    def test_ac1_stop_outside_recording_raises_invalid_phase(self) -> None:
        session = Session()

        with pytest.raises(InvalidPhase):
            session.stop(0.0)

    def test_ac1_report_before_finished_raises_invalid_phase(self) -> None:
        session = Session()

        with pytest.raises(InvalidPhase):
            session.report()

    def test_ac1_recording_status_outside_recording_raises_invalid_phase(self) -> None:
        session = Session()

        with pytest.raises(InvalidPhase):
            session.recording_status(0.0)

    def test_ac1_setup_readiness_while_recording_raises_invalid_phase(self) -> None:
        session = Session()
        session.start(0.0)

        with pytest.raises(InvalidPhase):
            session.setup_readiness(1.0)


class TestAC2StartGate:
    """AC2: observes with id-set {1,2,3} from t=10 to t=13 ->
    readiness reports three detections stable for 3 s; one observe with {1,2}
    resets stability to 0."""

    def test_ac2_stable_id_set_reports_count_and_duration_since_it_formed(
        self,
    ) -> None:
        session = Session()
        session.observe(10.0, frozenset({1, 2, 3}))
        session.observe(11.0, frozenset({1, 2, 3}))
        session.observe(12.0, frozenset({1, 2, 3}))
        session.observe(13.0, frozenset({1, 2, 3}))

        status = session.setup_readiness(13.0)
        assert (status.detected_count, status.stable_for_s) == (3, 3.0)

    def test_ac2_id_set_change_resets_stability_to_zero(self) -> None:
        session = Session()
        session.observe(10.0, frozenset({1, 2, 3}))
        session.observe(13.0, frozenset({1, 2, 3}))

        session.observe(14.0, frozenset({1, 2}))

        status = session.setup_readiness(14.0)
        assert (status.detected_count, status.stable_for_s) == (2, 0.0)


class TestAC3SteadyPresenceNoAbsences:
    """AC3: steady presence, no absences -> every instrument "present",
    with no usage windows in the final report."""

    def test_ac3_report_shows_every_instrument_present_with_empty_usage(
        self,
    ) -> None:
        session = Session()
        session.start(100.0)
        _confirm_presence(session, frozenset({1, 2, 3}), 101.0, 103.0)

        report = session.stop(110.0)

        assert {ir.tracker_id for ir in report.instruments} == {1, 2, 3}
        for instrument in report.instruments:
            assert instrument.completeness == "present"
            assert instrument.usage == ()
            assert instrument.label == f"Instrument {instrument.tracker_id}"

    def test_ac3_live_status_shows_only_identity_and_table_state(self) -> None:
        session = Session()
        session.start(100.0)
        _confirm_presence(session, frozenset({1, 2, 3}), 101.0, 103.0)

        _, instruments = session.recording_status(103.0)

        assert {inst.tracker_id for inst in instruments} == {1, 2, 3}
        for inst in instruments:
            assert inst.on_table is True
            assert set(inst.__dataclass_fields__) == {"tracker_id", "label", "on_table"}


class TestAC4ScriptedPickupAtNonZeroOrigin:
    """AC4: start(100), instrument absent for absolute t in (120,135) (last
    seen 120, back at 135) -> report window is SESSION-relative (off_s~20,
    on_s~35), not absolute; live status exposes table state but withholds
    analytics until the final report."""

    START_T = 100.0
    OTHERS = frozenset({2, 3})
    TARGET = 1
    ALL_PRESENT = frozenset({1, 2, 3})

    def _session_after_confirmed_absence(self) -> Session:
        session = Session()
        session.start(self.START_T)
        session.observe(101.0, self.ALL_PRESENT)  # recording begins
        session.observe(120.0, self.ALL_PRESENT)  # last seen before leaving
        session.observe(121.6, self.OTHERS)  # absence confirmed (>1.5s since 120)
        return session

    def _fully_scripted_session(self) -> Session:
        session = self._session_after_confirmed_absence()
        session.observe(135.0, self.ALL_PRESENT)  # first frame back
        session.observe(136.5, self.ALL_PRESENT)  # return confirmed (>1.0s since 135)
        return session

    def test_ac4_report_window_is_session_relative_not_absolute(self) -> None:
        session = self._fully_scripted_session()

        report = session.stop(140.0)

        candidates = [ir for ir in report.instruments if ir.tracker_id == self.TARGET]
        assert candidates
        target = candidates[0]
        assert len(target.usage) == 1
        assert target.usage[0].off_s == pytest.approx(20.0, abs=0.05)
        assert target.usage[0].on_s == pytest.approx(35.0, abs=0.05)

    def test_ac4_live_status_during_absence_shows_off_table_state(
        self,
    ) -> None:
        session = self._session_after_confirmed_absence()

        _, instruments = session.recording_status(125.0)

        candidates = [i for i in instruments if i.tracker_id == self.TARGET]
        assert candidates
        target = candidates[0]
        assert target.on_table is False


class TestAC5DebounceFlicker:
    """AC5: an absence of 0.9s (< off_debounce 1.5) leaves NO window; a
    presence blip of 0.5s during a real absence
    (< on_debounce 1.0) does NOT close the window."""

    def test_ac5_short_absence_below_off_debounce_leaves_no_trace(self) -> None:
        session = Session()
        session.start(0.0)
        _confirm_presence(session, frozenset({1}), 1.0, 2.5)
        session.observe(5.0, frozenset({1}))  # last seen before the blip
        session.observe(5.4, frozenset())  # reported absent
        session.observe(5.9, frozenset({1}))  # back; total absence 0.9s < 1.5

        _, instruments = session.recording_status(5.9)
        candidates = [i for i in instruments if i.tracker_id == 1]
        assert candidates
        assert candidates[0].on_table is True

        report = session.stop(10.0)
        report_candidates = [ir for ir in report.instruments if ir.tracker_id == 1]
        assert report_candidates
        assert report_candidates[0].completeness == "present"
        assert report_candidates[0].usage == ()

    def test_ac5_presence_blip_below_on_debounce_does_not_close_open_window(
        self,
    ) -> None:
        session = Session()
        session.start(0.0)
        _confirm_presence(session, frozenset({1}), 1.0, 2.5)
        session.observe(10.0, frozenset({1}))  # last seen before the real absence
        session.observe(12.0, frozenset())  # confirmed off (2.0s > 1.5)
        session.observe(12.3, frozenset({1}))  # brief return blip starts
        session.observe(12.8, frozenset())  # blip ends after 0.5s (< 1.0) -> not confirmed

        report = session.stop(15.0)

        candidates = [ir for ir in report.instruments if ir.tracker_id == 1]
        assert candidates
        instrument = candidates[0]
        assert instrument.completeness == "missing"
        assert len(instrument.usage) == 1
        assert instrument.usage[0].off_s == pytest.approx(10.0, abs=0.05)
        assert instrument.usage[0].on_s is None


class TestAC6NeverReturns:
    """AC6: id last seen at t=50, absent for all t>50 through stop(80) ->
    "missing", last window (50, None); report invariants hold (sorted,
    missing iff last window's on_s is None)."""

    def test_ac6_missing_completeness_open_window_and_report_invariants(
        self,
    ) -> None:
        session = Session()
        session.start(0.0)
        _confirm_presence(session, frozenset({1, 2}), 1.0, 2.5)
        session.observe(50.0, frozenset({1, 2}))  # id1 last seen
        session.observe(52.0, frozenset({2}))  # id1 confirmed off; id2 stays present

        report = session.stop(80.0)

        ids = [ir.tracker_id for ir in report.instruments]
        assert ids == sorted(ids)
        by_id = {ir.tracker_id: ir for ir in report.instruments}
        assert set(by_id) == {1, 2}

        missing = by_id[1]
        assert missing.completeness == "missing"
        assert len(missing.usage) == 1
        assert missing.usage[0].off_s == pytest.approx(50.0, abs=0.05)
        assert missing.usage[0].on_s is None

        present = by_id[2]
        assert present.completeness == "present"
        assert present.usage == ()

        for ir in report.instruments:
            is_missing = bool(ir.usage) and ir.usage[-1].on_s is None
            assert (ir.completeness == "missing") == is_missing


class TestAC6bBlinkAtStop:
    """AC6b: an instrument present throughout, whose id is absent only for
    the final <off_debounce_s before stop(t), stays "present" with usage==()
    and no open window -- the sub-debounce blip leaves no trace even though
    it touches Stop."""

    def test_ac6b_sub_debounce_blip_touching_stop_leaves_no_trace(self) -> None:
        session = Session()
        session.start(0.0)
        _confirm_presence(session, frozenset({1}), 1.0, 2.5)
        session.observe(79.7, frozenset({1}))  # last seen before the blink
        session.observe(79.9, frozenset())  # reported gone (blink starts)

        report = session.stop(80.0)  # 0.3s absence < 1.5 debounce at Stop's own t

        candidates = [ir for ir in report.instruments if ir.tracker_id == 1]
        assert candidates
        instrument = candidates[0]
        assert instrument.completeness == "present"
        assert instrument.usage == ()


class TestAC7MidRecordingEntryAndSpuriousId:
    """AC7: an id appearing mid-recording and continuously present
    > on_debounce_s joins the universe and can itself be picked up and
    reported; an id whose entire presence run is <= on_debounce_s before
    vanishing appears in neither live status nor the report, and does not
    corrupt others' windows."""

    def test_ac7_id_appearing_mid_recording_can_join_and_be_picked_up(
        self,
    ) -> None:
        session = Session()
        session.start(0.0)
        _confirm_presence(session, frozenset({1}), 1.0, 2.5)
        session.observe(10.0, frozenset({1, 9}))  # id9 appears mid-recording
        session.observe(11.5, frozenset({1, 9}))  # id9 confirmed (1.5s > 1.0)
        session.observe(20.0, frozenset({1, 9}))  # id9 last seen before its pickup
        session.observe(22.0, frozenset({1}))  # id9 confirmed off (2.0s > 1.5)

        report = session.stop(30.0)

        by_id = {ir.tracker_id: ir for ir in report.instruments}
        assert set(by_id) == {1, 9}
        assert by_id[9].completeness == "missing"
        assert len(by_id[9].usage) == 1
        assert by_id[9].usage[0].off_s == pytest.approx(20.0, abs=0.05)
        assert by_id[9].usage[0].on_s is None
        assert by_id[1].completeness == "present"
        assert by_id[1].usage == ()

    def test_ac7_spurious_id_below_on_debounce_never_enters_universe(self) -> None:
        session = Session()
        session.start(0.0)
        _confirm_presence(session, frozenset({1}), 1.0, 2.5)
        session.observe(10.0, frozenset({1, 99}))  # id99 flickers in
        session.observe(10.5, frozenset({1}))  # id99 gone after 0.5s (<= 1.0)

        _, instruments = session.recording_status(10.5)
        assert {i.tracker_id for i in instruments} == {1}

        report = session.stop(20.0)
        by_id = {ir.tracker_id: ir for ir in report.instruments}
        assert set(by_id) == {1}
        assert by_id[1].completeness == "present"
        assert by_id[1].usage == ()


class TestAC8RestartAfterFinishedResetsState:
    """AC8: start after FINISHED discards the old report (report() raises
    until the next stop), and debounce/pickup state is fully reset."""

    def test_ac8_new_start_discards_old_report_and_resets_debounce_state(
        self,
    ) -> None:
        session = Session()
        session.start(0.0)
        _confirm_presence(session, frozenset({1}), 1.0, 2.5)
        old_report = session.stop(5.0)
        assert old_report.instruments  # sanity: the old report is non-trivial

        session.start(6.0)

        with pytest.raises(InvalidPhase):
            session.report()

        # If debounce/pickup state weren't reset, id1 (already long-confirmed
        # in the previous recording) might immediately count as present here
        # too. It must instead re-earn entry from scratch.
        session.observe(6.5, frozenset({1}))  # only 0.5s into the new recording
        new_report = session.stop(7.0)

        assert new_report.instruments == ()


class TestAC9MonotonicGuardAndSetupIsolation:
    """AC9: non-monotonic observe raises ValueError; start/stop with
    t < last observe also raise; observe in SETUP never creates usage
    state."""

    def test_ac9_observe_with_earlier_t_raises_value_error(self) -> None:
        session = Session()
        session.observe(5.0, frozenset({1}))

        with pytest.raises(ValueError):
            session.observe(4.0, frozenset({1}))

    def test_ac9_observe_with_repeated_t_raises_value_error(self) -> None:
        session = Session()
        session.observe(5.0, frozenset({1}))

        with pytest.raises(ValueError):
            session.observe(5.0, frozenset({1}))

    def test_ac9_start_before_last_observe_raises_value_error(self) -> None:
        session = Session()
        session.observe(10.0, frozenset({1}))

        with pytest.raises(ValueError):
            session.start(5.0)

    def test_ac9_stop_before_last_observe_raises_value_error(self) -> None:
        session = Session()
        session.start(0.0)
        session.observe(10.0, frozenset({1}))

        with pytest.raises(ValueError):
            session.stop(5.0)

    def test_ac9_stop_equal_to_last_observe_is_allowed(self) -> None:
        session = Session()
        session.start(0.0)
        session.observe(5.0, frozenset({1}))

        report = session.stop(5.0)  # t == last observe, non-strict >= is fine

        assert report.duration_s == pytest.approx(5.0, abs=0.05)

    def test_ac9_setup_observes_never_seed_recording_usage_state(self) -> None:
        session = Session()
        session.observe(0.0, frozenset({1}))
        session.observe(19.0, frozenset({1}))  # present 19s, still in SETUP

        session.start(20.0)
        session.observe(20.5, frozenset({1}))  # only 0.5s into RECORDING (< 1.0)

        report = session.stop(21.0)

        assert report.duration_s == pytest.approx(1.0, abs=0.05)
        assert report.instruments == ()  # id1 never confirmed within RECORDING


class TestAC10NonMutatingProjection:
    """AC10: setup_readiness/recording_status project the debounced state at a
    queried t without committing anything -- polling never perturbs durable
    state, and a later stop() yields the same report either way."""

    def test_ac10_setup_readiness_projects_growing_stability_with_no_new_observes(
        self,
    ) -> None:
        session = Session()
        session.observe(5.0, frozenset({1, 2, 3}))

        assert session.setup_readiness(6.0).stable_for_s == 1.0
        assert session.setup_readiness(10.0).stable_for_s == 5.0

    def test_ac10_recording_status_projects_off_flip_with_no_intervening_observe(
        self,
    ) -> None:
        session = Session()
        session.start(0.0)
        _confirm_presence(session, frozenset({1}), 0.5, 2.0)
        session.observe(3.0, frozenset())  # absent; only 1.0s since last seen (2.0)

        _, early = session.recording_status(3.0)
        early_candidates = [i for i in early if i.tracker_id == 1]
        assert early_candidates
        assert early_candidates[0].on_table is True  # not yet past 1.5s off_debounce

        _, later = session.recording_status(4.0)  # 2.0s since last seen, no new observe
        later_candidates = [i for i in later if i.tracker_id == 1]
        assert later_candidates
        assert later_candidates[0].on_table is False

    def test_ac10_polling_accessors_before_stop_does_not_perturb_the_report(
        self,
    ) -> None:
        def build() -> Session:
            session = Session()
            session.start(0.0)
            _confirm_presence(session, frozenset({1}), 0.5, 2.0)
            session.observe(3.0, frozenset())
            return session

        unpolled = build()
        polled = build()
        polled.recording_status(3.5)
        polled.recording_status(4.0)
        polled.recording_status(4.6)

        report_unpolled = unpolled.stop(5.0)
        report_polled = polled.stop(5.0)

        assert report_polled == report_unpolled
        assert report_polled.instruments  # sanity: not vacuously equal-empty


class TestAC11FinishedGateAndReportImmutability:
    """AC11: D15 finished-phase gate + report immutability -- fresh session
    readiness is empty before any observe; post-stop observes keep driving its
    count/stability projection (a held id-set keeps growing, a changed one
    resets) while report() stays byte-identical."""

    def test_ac11_fresh_gate_is_zero_and_post_stop_observes_still_drive_it(
        self,
    ) -> None:
        session = Session()
        initial = session.setup_readiness(0.0)
        assert (initial.detected_count, initial.stable_for_s) == (0, 0.0)

        session.start(0.0)
        _confirm_presence(session, frozenset({1}), 1.0, 2.5)
        session.stop(10.0)

        session.observe(11.0, frozenset({5}))  # new id-set, post-stop (FINISHED)
        status = session.setup_readiness(11.0)
        assert (status.detected_count, status.stable_for_s) == (1, 0.0)

        session.observe(12.0, frozenset({5}))  # held -> stability grows
        status = session.setup_readiness(12.0)
        assert (status.detected_count, status.stable_for_s) == (1, 1.0)

        session.observe(13.0, frozenset({5, 6}))  # changed -> resets
        status = session.setup_readiness(13.0)
        assert (status.detected_count, status.stable_for_s) == (2, 0.0)

    def test_ac11_report_is_byte_identical_before_and_after_post_stop_observes(
        self,
    ) -> None:
        session = Session()
        session.start(0.0)
        _confirm_presence(session, frozenset({1}), 1.0, 2.5)
        report_at_stop = session.stop(10.0)

        session.observe(11.0, frozenset({5}))
        session.observe(12.0, frozenset({5}))

        assert session.report() == report_at_stop
        assert report_at_stop.instruments  # sanity: comparison isn't vacuous


def _recorded_ids(session: Session, t: float) -> set[int]:
    _, instruments = session.recording_status(t)
    return {instrument.tracker_id for instrument in instruments}


class TestBS1RecordingIsFilteredByTheRoster:
    """B-S1: only the RECORDING half of `observe` changes. It uses
    `present_ids & roster`, so a not-in-roster id never becomes a track, never
    confirms, and never reaches `recording_status()` or the report. The Start
    gate (`setup_readiness`) keeps using the FULL
    `present_ids` — it is the operator's judgment on everything detected."""

    def test_b_s1_a_not_in_roster_id_never_enters_the_live_recording_status(
        self,
    ) -> None:
        session = Session()
        session.start(0.0)
        session.observe(1.0, frozenset({1, 9}), frozenset({1}))
        session.observe(3.0, frozenset({1, 9}), frozenset({1}))  # 2.0s > 1.0 debounce

        assert _recorded_ids(session, 3.0) == {1}

    def test_b_s1_a_not_in_roster_id_never_reaches_the_report(self) -> None:
        session = Session()
        session.start(0.0)
        session.observe(1.0, frozenset({1, 9}), frozenset({1}))
        session.observe(3.0, frozenset({1, 9}), frozenset({1}))
        session.observe(20.0, frozenset({1, 9}), frozenset({1}))

        report = session.stop(30.0)

        assert {ir.tracker_id for ir in report.instruments} == {1}

    def test_b_s1_a_not_in_roster_id_never_enters_the_live_roster(
        self,
    ) -> None:
        # A foreign object must not enter the rows shown during recording.
        session = Session()
        session.start(0.0)
        roster = frozenset({1, 2, 3})
        session.observe(1.0, frozenset({1, 2, 3, 9}), roster)
        session.observe(3.0, frozenset({1, 2, 3, 9}), roster)

        _, instruments = session.recording_status(3.0)

        assert {instrument.tracker_id for instrument in instruments} == {1, 2, 3}

    def test_b_s1_a_not_in_roster_id_leaving_creates_no_usage_window(self) -> None:
        session = Session()
        session.start(0.0)
        roster = frozenset({1})
        session.observe(1.0, frozenset({1, 9}), roster)
        session.observe(3.0, frozenset({1, 9}), roster)
        session.observe(10.0, frozenset({1, 9}), roster)  # id 9 last seen
        session.observe(12.0, frozenset({1}), roster)  # ... and gone for good

        report = session.stop(20.0)

        by_id = {ir.tracker_id: ir for ir in report.instruments}
        assert set(by_id) == {1}
        assert by_id[1].completeness == "present"
        assert by_id[1].usage == ()  # id 9's exit did not smear onto id 1

    def test_b_s1_the_start_gate_still_counts_every_detected_id(self) -> None:
        # The roster is the LINKER's judgment about a recording; the Start gate
        # is the OPERATOR's judgment about the table. A foreign object sitting
        # on the tray must still show up in the count they are looking at.
        session = Session()
        session.observe(10.0, frozenset({1, 2, 9}), frozenset({1, 2}))
        session.observe(11.0, frozenset({1, 2, 9}), frozenset({1, 2}))

        status = session.setup_readiness(11.0)
        assert (status.detected_count, status.stable_for_s) == (3, 1.0)

    def test_b_s1_the_id_set_stability_clock_still_watches_every_detected_id(
        self,
    ) -> None:
        session = Session()
        roster = frozenset({1, 2})
        session.observe(10.0, frozenset({1, 2}), roster)
        session.observe(11.0, frozenset({1, 2}), roster)

        session.observe(12.0, frozenset({1, 2, 9}), roster)  # a foreign id lands

        status = session.setup_readiness(12.0)
        assert (status.detected_count, status.stable_for_s) == (3, 0.0)


class TestBS2RosterDefaultsToUnfiltered:
    """B-S2: `roster=None` (the default) preserves today's unfiltered
    behaviour, so existing callers and tests are untouched semantically."""

    def test_b_s2_omitting_the_roster_lets_every_present_id_confirm(self) -> None:
        session = Session()
        session.start(0.0)
        session.observe(1.0, frozenset({1, 9}))
        session.observe(3.0, frozenset({1, 9}))

        assert _recorded_ids(session, 3.0) == {1, 9}

    def test_b_s2_an_explicit_none_roster_is_the_same_as_omitting_it(self) -> None:
        session = Session()
        session.start(0.0)
        session.observe(1.0, frozenset({1, 9}), None)
        session.observe(3.0, frozenset({1, 9}), None)

        assert _recorded_ids(session, 3.0) == {1, 9}


class TestBS3SessionStaysPure:
    """B-S3 (D11): the roster arrives as data, per call — the session holds no
    clock, no thread, and no tracker reference, so it can neither look the
    roster up nor remember the last one it saw."""

    def test_b_s3_the_roster_filters_per_call_not_retroactively(self) -> None:
        # An id that joins the roster late starts its entry debounce THEN — the
        # frames before it were filtered out, so they cannot count toward it.
        session = Session()
        session.start(0.0)
        session.observe(1.0, frozenset({1, 5}), frozenset({1}))  # 5 filtered out
        session.observe(2.0, frozenset({1, 5}), frozenset({1, 5}))  # 5's run starts

        assert _recorded_ids(session, 2.8) == {1}  # only 0.8s in: not yet confirmed

        session.observe(3.5, frozenset({1, 5}), frozenset({1, 5}))  # 1.5s > 1.0
        assert _recorded_ids(session, 3.5) == {1, 5}

    def test_b_s3_a_roster_is_not_remembered_by_the_next_observe(self) -> None:
        session = Session()
        session.start(0.0)
        session.observe(1.0, frozenset({1, 9}), frozenset({1}))  # 9 filtered
        session.observe(2.0, frozenset({1, 9}))  # no roster: 9's run starts here
        session.observe(3.5, frozenset({1, 9}))  # 1.5s > 1.0 -> 9 confirms

        assert _recorded_ids(session, 3.5) == {1, 9}

    def test_b_s3_the_same_script_yields_the_same_report_every_time(self) -> None:
        def run() -> object:
            session = Session()
            session.start(0.0)
            roster = frozenset({1, 2})
            session.observe(1.0, frozenset({1, 2, 9}), roster)
            session.observe(3.0, frozenset({1, 2, 9}), roster)
            session.observe(10.0, frozenset({2, 9}), roster)
            session.observe(12.0, frozenset({2, 9}), roster)
            return session.stop(20.0)

        first, second = run(), run()

        assert first == second
        assert first.instruments  # sanity: not vacuously equal-empty


class TestBS5EmptyRosterWhileRecording:
    """B-S5: defensive recording behavior if a producer temporarily supplies an
    empty roster. Start normally preserves an already-approved full roster, but
    the session filter must still admit nothing from an explicit empty set."""

    def test_b_s5_an_empty_roster_confirms_nothing_while_it_lasts(self) -> None:
        session = Session()
        session.start(0.0)
        session.observe(1.0, frozenset({1, 2, 3}), frozenset())
        session.observe(5.0, frozenset({1, 2, 3}), frozenset())  # 4s, way past debounce

        _, instruments = session.recording_status(5.0)

        assert instruments == ()

    def test_b_s5_a_short_empty_roster_window_leaves_the_report_unchanged(self) -> None:
        def run(initial_roster: frozenset[int]) -> object:
            session = Session()
            session.start(0.0)
            roster = frozenset({1, 2})
            # A defensive short producer glitch, then the approved roster returns.
            session.observe(0.1, frozenset({1, 2}), initial_roster)
            session.observe(0.4, frozenset({1, 2}), initial_roster)
            session.observe(0.7, frozenset({1, 2}), initial_roster)
            session.observe(1.0, frozenset({1, 2}), roster)
            session.observe(3.0, frozenset({1, 2}), roster)
            session.observe(20.0, frozenset({1}), roster)  # id 2 walks off
            session.observe(22.0, frozenset({1}), roster)
            return session.stop(30.0)

        with_empty_window = run(frozenset())
        without = run(frozenset({1, 2}))

        assert with_empty_window == without
        assert {ir.tracker_id for ir in without.instruments} == {1, 2}  # sanity


class TestStaleAccessorsDoNotRewindDurableState:
    """General semantics (Monotonic t guard): a read-only accessor called
    with a t smaller than the last known t returns the state as of that last
    known t (it doesn't rewind, and it doesn't raise). Supports AC9/AC10's
    non-mutating-accessor guarantees; not itself a numbered AC."""

    def test_setup_readiness_with_a_stale_t_returns_the_last_known_state(self) -> None:
        session = Session()
        session.observe(10.0, frozenset({1, 2, 3}))
        session.observe(13.0, frozenset({1, 2, 3}))

        result = session.setup_readiness(5.0)  # smaller than last observe (13.0)

        assert (result.detected_count, result.stable_for_s) == (3, 3.0)


# --- T11 B2: pure setup readiness (catalog / roster / resolving verdict) -------

_STABLE_S = 2.0  # the readiness stability threshold these tests construct with


class TestReadinessBlockingReasons:
    """T11/B2 backend test 2: `setup_readiness` is a pure verdict on the latest
    same-tick observation. The gate is permissive: a subset of the catalog may
    start and settled Unknown objects never block. It is NOT ready — with the
    right priority-ordered blocking reason — while a track is still resolving, no
    instrument is recognised yet, or the id-set has not held steady for
    `setup_stable_s`. Priority: recognising > missing_instruments > hold_steady."""

    def test_nothing_observed_yet_is_not_ready_and_recognising(self) -> None:
        session = Session(setup_stable_s=_STABLE_S)
        readiness = session.setup_readiness(1.0)
        assert readiness == SetupReadiness(0, 0, 0, 0, 0, 0.0, False, "recognising")

    def test_a_resolving_track_blocks_with_recognising(self) -> None:
        session = Session(setup_stable_s=_STABLE_S)
        catalog = frozenset({1, 2, 3})
        # Whole catalog present and long-settled, but id 2 is still resolving.
        session.observe(0.0, catalog, catalog, catalog, frozenset({2}))

        readiness = session.setup_readiness(_STABLE_S + 3.0)

        assert readiness.ready is False
        assert readiness.blocking_reason == "recognising"
        assert readiness.resolving_count == 1

    def test_a_partial_catalog_subset_is_ready(self) -> None:
        # A subset of the catalog (2 of 3) is now a valid start: the operator can
        # lay out and record just the instruments they care about.
        session = Session(setup_stable_s=_STABLE_S)
        catalog = frozenset({1, 2, 3})
        present = frozenset({1, 2})  # id 3 (a catalog member) is not on the tray
        session.observe(0.0, present, present, catalog, frozenset())

        readiness = session.setup_readiness(_STABLE_S + 3.0)

        assert readiness.ready is True
        assert readiness.blocking_reason is None
        assert readiness.recognised_count == 2
        assert readiness.expected_count == 3

    def test_nothing_recognised_blocks_with_missing_instruments(self) -> None:
        # `missing_instruments` now means "no recognised instrument yet" -- a tray
        # holding only a non-roster object still needs at least one known one.
        session = Session(setup_stable_s=_STABLE_S)
        catalog = frozenset({1, 2, 3})
        present = frozenset({9})  # only a non-roster object, nothing recognised
        session.observe(0.0, present, frozenset(), catalog, frozenset())

        readiness = session.setup_readiness(_STABLE_S + 3.0)

        assert readiness.ready is False
        assert readiness.blocking_reason == "missing_instruments"
        assert readiness.recognised_count == 0

    def test_a_settled_unknown_object_no_longer_blocks(self) -> None:
        # A settled Unknown on the tray is reported (unknown_count == 1) but does
        # NOT block Start -- it is simply excluded from the roster at Start.
        session = Session(setup_stable_s=_STABLE_S)
        catalog = frozenset({1, 2, 3})
        roster = frozenset({1, 2, 3})
        present = frozenset({1, 2, 3, 9})  # id 9 is non-roster and NOT resolving
        session.observe(0.0, present, roster, catalog, frozenset())

        readiness = session.setup_readiness(_STABLE_S + 3.0)

        assert readiness.ready is True
        assert readiness.blocking_reason is None
        assert readiness.unknown_count == 1

    def test_full_catalog_but_not_yet_stable_blocks_with_hold_steady(self) -> None:
        session = Session(setup_stable_s=_STABLE_S)
        catalog = frozenset({1, 2, 3})
        session.observe(0.0, catalog, catalog, catalog, frozenset())

        # Queried BEFORE setup_stable_s has elapsed — everything is recognised and
        # settled, but the id-set has not held long enough yet.
        readiness = session.setup_readiness(_STABLE_S - 0.5)

        assert readiness.ready is False
        assert readiness.blocking_reason == "hold_steady"
        assert readiness.stable_for_s == pytest.approx(_STABLE_S - 0.5)

    def test_the_stability_threshold_is_the_constructor_param(self) -> None:
        # The same observation is hold_steady under a strict threshold and ready
        # under a lax one — the threshold is the injected param, not a constant.
        catalog = frozenset({1, 2})

        strict = Session(setup_stable_s=5.0)
        strict.observe(0.0, catalog, catalog, catalog, frozenset())
        assert strict.setup_readiness(1.0).blocking_reason == "hold_steady"

        lax = Session(setup_stable_s=0.5)
        lax.observe(0.0, catalog, catalog, catalog, frozenset())
        assert lax.setup_readiness(1.0).ready is True

    def test_readiness_is_illegal_while_recording(self) -> None:
        session = Session(setup_stable_s=_STABLE_S)
        session.start(0.0)
        with pytest.raises(InvalidPhase):
            session.setup_readiness(1.0)


class TestReadinessAllowsSubsetAndIgnoresUnknowns:
    """T11/B2 backend test 3: readiness is True whenever at least one instrument
    is recognised, with no resolving track, held past `setup_stable_s`. A subset
    of the catalog is fine and an extra settled Unknown id does not flip it back
    to not-ready. Only zero recognised, a resolving track, an empty catalog, or
    insufficient stability keep it not-ready."""

    def test_exact_present_catalog_after_stability_is_ready(self) -> None:
        session = Session(setup_stable_s=_STABLE_S)
        catalog = frozenset({1, 2, 3})
        session.observe(0.0, catalog, catalog, catalog, frozenset())

        readiness = session.setup_readiness(_STABLE_S + 0.5)

        assert readiness.ready is True
        assert readiness.blocking_reason is None
        assert readiness.detected_count == 3
        assert readiness.expected_count == 3
        assert readiness.recognised_count == 3
        assert readiness.unknown_count == 0
        assert readiness.resolving_count == 0

    def test_one_extra_unknown_id_stays_ready(self) -> None:
        session = Session(setup_stable_s=0.0)
        catalog = frozenset({1, 2, 3})
        session.observe(0.0, catalog | {9}, catalog, catalog, frozenset())

        readiness = session.setup_readiness(5.0)

        assert readiness.ready is True
        assert readiness.blocking_reason is None
        assert readiness.unknown_count == 1

    def test_one_missing_catalog_id_stays_ready(self) -> None:
        session = Session(setup_stable_s=0.0)
        catalog = frozenset({1, 2, 3})
        present = frozenset({1, 2})
        session.observe(0.0, present, present, catalog, frozenset())

        readiness = session.setup_readiness(5.0)

        assert readiness.ready is True
        assert readiness.blocking_reason is None
        assert readiness.recognised_count == 2

    def test_an_empty_catalog_is_never_ready(self) -> None:
        # A guard on the `expected_count > 0` clause: with no catalog configured
        # the gate never opens (fail-closed), even with a settled tray.
        session = Session(setup_stable_s=0.0)
        session.observe(0.0, frozenset({1, 2}), frozenset({1, 2}), frozenset(), frozenset())

        readiness = session.setup_readiness(5.0)

        assert readiness.ready is False
        assert readiness.blocking_reason == "missing_instruments"
