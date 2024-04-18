from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class RunConfig:
    seed: int = field()
    epochs: int = field()
    layers: int = field()
    heads: int = field()
    embd: int = field()
    run_name: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        # unique identification of run - for dissertation anway, include year if multi-year experiment 
        if self.run_name is None:
            self.run_name = datetime.now().strftime("run_%d-%m_%H:%M:%S.%f")
        
        if self.seed is None:
            self.seed = 123

        if self.embd % self.heads != 0:
            raise ValueError("Embd dim should be divisible by number of heads!")
        