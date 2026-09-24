"""Debug-only collision filters. Isolation assets and official robots are not overwritten.

PhysX word2: shapes do not collide if (g0[2] & g1[2]) != 0.

BIT_FINGER_INNER: left fingers ignore deskbin inner/bottom (balls still hit inner).
BIT_LINK6_BIN: wrist/palm link fl_link6 ignores the whole deskbin.
Grasp slabs stay collidable with fingers.
"""
from __future__ import annotations

BIT_FINGER_INNER = 1 << 0
BIT_LINK6_BIN = 1 << 1
FINGER_LINKS = ("fl_link7", "fl_link8")
PALM_LINKS = ("fl_link6",)


def _or_word2(shape, bits: int) -> None:
    groups = [int(x) for x in shape.get_collision_groups()]
    groups[2] = int(groups[2]) | int(bits)
    shape.set_collision_groups(groups)


def _link_shapes(entity, names: tuple[str, ...]):
    shapes = []
    links = list(entity.get_links()) if hasattr(entity, "get_links") else []
    for link in links:
        if link.get_name() not in names:
            continue
        getter = getattr(link, "get_collision_shapes", None)
        if callable(getter):
            shapes.extend(list(getter()))
    return shapes


def apply_retention_collision_filters(task) -> dict:
    from patch_grasp_surface import deskbin_shapes

    _, deskbin_shapes_list, roles, names = deskbin_shapes(task.deskbin)
    inner_n = 0
    grasp_n = 0
    for shape, role in zip(deskbin_shapes_list, roles):
        if role == "grasp":
            _or_word2(shape, BIT_LINK6_BIN)
            grasp_n += 1
        else:
            _or_word2(shape, BIT_FINGER_INNER | BIT_LINK6_BIN)
            inner_n += 1
    finger_shapes = _link_shapes(task.robot.left_entity, FINGER_LINKS)
    palm_shapes = _link_shapes(task.robot.left_entity, PALM_LINKS)
    for shape in finger_shapes:
        _or_word2(shape, BIT_FINGER_INNER)
    for shape in palm_shapes:
        _or_word2(shape, BIT_LINK6_BIN)
    receipt = {
        "scope": "debug_not_official",
        "deskbin_grasp_shapes": grasp_n,
        "deskbin_inner_bottom_shapes": inner_n,
        "finger_shapes": len(finger_shapes),
        "palm_shapes": len(palm_shapes),
        "rule": "fingers ignore inner/bottom; fl_link6 ignores entire deskbin; grasp slabs remain",
        "bits": {"BIT_FINGER_INNER": BIT_FINGER_INNER, "BIT_LINK6_BIN": BIT_LINK6_BIN},
    }
    task._af_retention_collision_filter = receipt
    if len(finger_shapes) < 2:
        raise RuntimeError(f"expected >=2 finger collision shapes, got {len(finger_shapes)}")
    if grasp_n < 1 or inner_n < 1:
        raise RuntimeError("deskbin split roles missing")
    return receipt
