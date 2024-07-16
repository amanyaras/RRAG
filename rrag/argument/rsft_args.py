from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

from typing_extensions import Self


if TYPE_CHECKING:
    import torch


@dataclass
class RetrivModelSFTArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
