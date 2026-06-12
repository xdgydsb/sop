"""Executable SOP examples used by tests and early adapters."""

from sop_contracts import (
    FactCondition,
    FactOperator,
    OperationDefinition,
    SopDefinition,
    TransitionCondition,
)


def package_five_step_sop() -> SopDefinition:
    inside = lambda key: FactCondition(key, FactOperator.EQUALS, "inside")
    return SopDefinition(
        sop_id="package-five-step",
        version=1,
        operations=(
            OperationDefinition(
                operation_id="open_box",
                name="打开盒子",
                trigger=TransitionCondition(
                    "box.state", "closed", "open", interaction="hand:box"
                ),
                preconditions=(
                    FactCondition("box.state", FactOperator.EQUALS, "closed"),
                ),
                postconditions=(
                    FactCondition("box.state", FactOperator.EQUALS, "open"),
                ),
                hold_ms=300,
            ),
            OperationDefinition(
                operation_id="place_earphone",
                name="放入耳机",
                prerequisites=("open_box",),
                trigger=TransitionCondition(
                    "earphone.location",
                    "outside",
                    "inside",
                    interaction="hand:earphone",
                ),
                preconditions=(
                    FactCondition(
                        "earphone.location", FactOperator.EQUALS, "outside"
                    ),
                ),
                postconditions=(inside("earphone.location"),),
                hold_ms=300,
            ),
            OperationDefinition(
                operation_id="place_charger",
                name="放入充电器",
                prerequisites=("open_box", "place_earphone"),
                trigger=TransitionCondition(
                    "charger.location",
                    "outside",
                    "inside",
                    interaction="hand:charger",
                ),
                preconditions=(
                    FactCondition("charger.location", FactOperator.EQUALS, "outside"),
                ),
                postconditions=(inside("charger.location"),),
                hold_ms=300,
            ),
            OperationDefinition(
                operation_id="place_green_bag",
                name="放入绿色小袋",
                prerequisites=("open_box", "place_earphone", "place_charger"),
                trigger=TransitionCondition(
                    "green_bag.location",
                    "outside",
                    "inside",
                    interaction="hand:green_bag",
                ),
                preconditions=(
                    FactCondition(
                        "green_bag.location", FactOperator.EQUALS, "outside"
                    ),
                ),
                postconditions=(inside("green_bag.location"),),
                hold_ms=300,
            ),
            OperationDefinition(
                operation_id="close_box",
                name="关闭盒子",
                prerequisites=(
                    "open_box",
                    "place_earphone",
                    "place_charger",
                    "place_green_bag",
                ),
                trigger=TransitionCondition(
                    "box.state", "open", "closed", interaction="hand:box"
                ),
                preconditions=(
                    inside("earphone.location"),
                    inside("charger.location"),
                    inside("green_bag.location"),
                ),
                postconditions=(
                    FactCondition("box.state", FactOperator.EQUALS, "closed"),
                ),
                hold_ms=300,
            ),
        ),
    )
