import torch 
from typing import List, Optional

class PullKey:
    pd_pair: Optional[List[int]]
    input_tokens: Optional[List[torch.Tensor]]
    ending_signal: Optional[bool]
    
    def __init__(self, pd_pair: Optional[List[int]] = None, 
                 input_tokens: Optional[List[torch.Tensor]] = None,
                 ending_signal :Optional[bool] = None):
        self.pd_pair = pd_pair
        self.input_tokens = input_tokens
        self.ending_signal = ending_signal
    