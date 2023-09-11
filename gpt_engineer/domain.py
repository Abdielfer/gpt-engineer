from typing import Callable, List, TypeVar

from ai import AI
from db import DBs

Step = TypeVar("Step", bound=Callable[[AI, DBs], List[dict]])
