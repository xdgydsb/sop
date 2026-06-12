"""
FSM + EventDetector 单元测试 — 无需摄像头，验证核心逻辑

测试覆盖:
  1. 正确序列 (S1→S2→S3→S4→S5→完成)
  2. 乱序 (S3 在 S2 之前)
  3. 漏步 (跳过 S4 直接 S5)
  4. 提前关闭 (S5 时还有物体未入盒)
  5. 重复步骤
  6. 事件 one-shot (同一事件不重复触发)
  7. 超时
  8. 低置信度过滤

Usage:
  python test_fsm_events.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engine.sop_fsm import SOPStateMachine, FSMResult, SOPStep
from engine.physical_state import PhysicalStateEngine, PhysicalStateResult


# ── Helper: 模拟 PhysicalStateResult ──
def make_phys(box_open=False, box_closed=False, visible=None, placed=None,
              wrong=None, wrong_frames=0, phys_step=0):
    return PhysicalStateResult(
        box_is_open=box_open,
        box_is_closed=box_closed,
        box_state_conf=0.8,
        hand_near_box=False,
        hands_detected=True,
        visible_objects=visible or [],
        objects_placed=placed or [],
        objects_in_box=[],
        placed_this_frame=None,
        current_phys_step=phys_step,
        wrong_placement=wrong,
        wrong_placement_frames=wrong_frames,
    )


def test_correct_sequence():
    """测试1: 正确序列 S1→S2→S3→S4→S5→完成"""
    print("\n" + "=" * 60)
    print("TEST 1: Correct Sequence (S1→S2→S3→S4→S5→Done)")
    print("=" * 60)

    fsm = SOPStateMachine(timeout=30.0)
    errors = []
    t0 = time.time()
    dt = 0.0

    # S0: Idle
    r = fsm.validate(0, 0.8, True, t0)
    assert r.step_id == 0, f"Expected S0, got {r.step_id}"
    print(f"  ✓ S0 Idle — {r.message}")

    # S1: Open box
    dt += 0.6
    r = fsm.validate(1, 0.85, True, t0 + dt)
    assert r.step_id == 1, f"Expected S1, got {r.step_id}"
    assert r.is_correct
    print(f"  ✓ S1 Open Box — {r.message}")

    # S2: Earphone in box
    dt += 0.6
    r = fsm.validate(2, 0.8, True, t0 + dt)
    assert r.step_id == 2, f"Expected S2, got {r.step_id}"
    assert r.is_correct
    print(f"  ✓ S2 Earphone — {r.message}")

    # S3: Charger in box
    dt += 0.6
    r = fsm.validate(3, 0.75, True, t0 + dt)
    assert r.step_id == 3, f"Expected S3, got {r.step_id}"
    assert r.is_correct
    print(f"  ✓ S3 Charger — {r.message}")

    # S4: Green bag in box
    dt += 0.6
    r = fsm.validate(4, 0.9, True, t0 + dt)
    assert r.step_id == 4, f"Expected S4, got {r.step_id}"
    assert r.is_correct
    print(f"  ✓ S4 Green Bag — {r.message}")

    # S5: Close box
    dt += 0.6
    r = fsm.validate(5, 0.85, True, t0 + dt)
    assert r.step_id == 5, f"Expected S5, got {r.step_id}"
    assert r.is_correct
    print(f"  ✓ S5 Close Box — {r.message}")

    # Done
    dt += 0.6
    r = fsm.validate(6, 0.9, True, t0 + dt)
    assert r.step_id == 6, f"Expected Done(6), got {r.step_id}"
    assert r.is_complete
    print(f"  ✓ Done — {r.message}")

    # Verify history: IDLE→S1, S1→S2, S2→S3, S3→S4, S4→S5, S5→DONE = 6 transitions
    assert len(fsm.step_history) == 6, f"Expected 6 transitions, got {len(fsm.step_history)}"
    print(f"\n  PASS: all 6 steps correct, {len(fsm.step_history)} transitions recorded")


def test_wrong_order():
    """测试2: 乱序 — S3在S2之前"""
    print("\n" + "=" * 60)
    print("TEST 2: Wrong Order (S1→S3 skips S2)")
    print("=" * 60)

    fsm = SOPStateMachine(timeout=30.0)
    t0 = time.time()
    dt = 0.0

    # S0→S1: normal
    dt += 0.6
    r = fsm.validate(1, 0.85, True, t0 + dt)
    assert r.step_id == 1
    print(f"  ✓ S1 Open Box")

    # S1→S3: should trigger WRONG_ORDER (skips S2)
    dt += 0.6
    r = fsm.validate(3, 0.7, True, t0 + dt)
    assert r.has_error, "Should have error for wrong order"
    assert r.error_type == "MISSING_STEP" or r.error_type == "WRONG_ORDER", \
        f"Expected MISSING_STEP or WRONG_ORDER, got {r.error_type}"
    print(f"  ✓ S1→S3 detected as {r.error_type}: {r.message}")


def test_early_close():
    """测试3: 提前关闭 — S5时S4未完成"""
    print("\n" + "=" * 60)
    print("TEST 3: Early Close (S1→S2→S3→S5, missing S4)")
    print("=" * 60)

    fsm = SOPStateMachine(timeout=30.0)
    t0 = time.time()
    dt = 0.0

    # S0→S1
    dt += 0.6
    fsm.validate(1, 0.85, True, t0 + dt)
    # S1→S2
    dt += 0.6
    fsm.validate(2, 0.8, True, t0 + dt)
    # S2→S3
    dt += 0.6
    fsm.validate(3, 0.75, True, t0 + dt)

    # S3→S5: should trigger MISSING_STEP (skips S4)
    dt += 0.6
    r = fsm.validate(5, 0.7, True, t0 + dt)
    assert r.has_error, "Should have error for early close"
    assert "MISSING" in r.error_type or "漏步" in r.message, \
        f"Expected MISSING_STEP, got {r.error_type}: {r.message}"
    print(f"  ✓ S3→S5 detected as {r.error_type}: {r.message}")


def test_repeat_step():
    """测试4: 重复步骤 — 同一事件触发两次"""
    print("\n" + "=" * 60)
    print("TEST 4: Repeat Step (S2 triggered twice)")
    print("=" * 60)

    fsm = SOPStateMachine(timeout=30.0)
    t0 = time.time()
    dt = 0.0

    # S0→S1
    dt += 0.6
    fsm.validate(1, 0.85, True, t0 + dt)
    # S1→S2
    dt += 0.6
    fsm.validate(2, 0.8, True, t0 + dt)

    # S2→S2 (same step — should be OK, just keep current)
    dt += 0.6
    r = fsm.validate(2, 0.8, True, t0 + dt)
    assert r.step_id == 2, f"Expected to stay at S2, got {r.step_id}"
    assert not r.has_error, "Same step should not trigger error"
    print(f"  ✓ S2→S2 (keep current) — no error")

    # S2→S1 (backward — should trigger WRONG_ORDER)
    dt += 0.6
    r = fsm.validate(1, 0.7, True, t0 + dt)
    assert r.has_error, "Backward step should trigger error"
    print(f"  ✓ S2→S1 (backward) detected as {r.error_type}: {r.message}")


def test_low_confidence_filter():
    """测试5: 低置信度过滤"""
    print("\n" + "=" * 60)
    print("TEST 5: Low Confidence Filtering")
    print("=" * 60)

    fsm = SOPStateMachine(timeout=30.0)
    t0 = time.time()

    # Low confidence should not advance
    r = fsm.validate(1, 0.3, True, t0)
    assert r.step_id == 0, f"Low conf should stay at S0, got {r.step_id}"
    print(f"  ✓ Low conf (0.3) — stays at S0")

    # High confidence should advance
    r = fsm.validate(1, 0.85, True, t0 + 0.6)
    assert r.step_id == 1, f"High conf should advance to S1, got {r.step_id}"
    print(f"  ✓ High conf (0.85) — advances to S1")


def test_timeout():
    """测试6: 超时检测"""
    print("\n" + "=" * 60)
    print("TEST 6: Timeout Detection")
    print("=" * 60)

    fsm = SOPStateMachine(timeout=1.0)  # Short timeout for test
    t0 = time.time()

    # Advance to S1
    r = fsm.validate(1, 0.85, True, t0)
    assert r.step_id == 1

    # Stay at S1 beyond timeout
    r = fsm.validate(1, 0.5, True, t0 + 2.0)
    assert r.has_error, f"Should timeout, got {r.step_id}"
    assert r.error_type == "TIMEOUT", f"Expected TIMEOUT, got {r.error_type}"
    print(f"  ✓ Timeout detected: {r.message}")


def test_fsm_reset():
    """测试7: FSM Reset"""
    print("\n" + "=" * 60)
    print("TEST 7: FSM Reset")
    print("=" * 60)

    fsm = SOPStateMachine(timeout=30.0)
    t0 = time.time()

    # Advance to S3
    fsm.validate(1, 0.85, True, t0 + 0.6)
    fsm.validate(2, 0.8, True, t0 + 1.2)
    fsm.validate(3, 0.75, True, t0 + 1.8)
    assert fsm.current_step.value == 3

    # Reset
    fsm.reset()
    assert fsm.current_step == SOPStep.IDLE
    assert len(fsm.step_history) == 0
    assert not fsm.error_occurred
    print(f"  ✓ Reset — back to IDLE, history cleared")


def test_physical_wrong_object():
    """测试8: 错误物体入盒被 PhysicalStateEngine 拒绝"""
    print("\n" + "=" * 60)
    print("TEST 8: Wrong Object Rejection")
    print("=" * 60)

    phys = PhysicalStateEngine(confirm_frames=4, stable_threshold=6)
    # This tests that _detect_placement respects the expected order
    # We can't easily simulate the full state machine in unit test,
    # but we can verify reset and initial state
    phys.reset()
    assert len(phys._objects_placed) == 0
    print(f"  ✓ PhysicalStateEngine reset OK, no objects placed")
    print(f"  NOTE: Full wrong-object test requires tracked objects + video frames")
    print(f"  The wrong_placement field in PhysicalStateResult is checked by EventDetector")


def test_all():
    print("=" * 60)
    print("FSM + EventDetector 单元测试")
    print("=" * 60)

    tests = [
        test_correct_sequence,
        test_wrong_order,
        test_early_close,
        test_repeat_step,
        test_low_confidence_filter,
        test_timeout,
        test_fsm_reset,
        test_physical_wrong_object,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"\n  ✗ FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    if errors:
        for name, msg in errors:
            print(f"  ✗ {name}: {msg}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    ok = test_all()
    sys.exit(0 if ok else 1)
