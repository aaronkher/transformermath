import torch

class RunDevice:

    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            
        else:
            self.device = torch.device("cpu")
        print(f"|Device: {self.device}|")

    def get_device(self) -> torch.device:
        return self.device
    
    def set_device(self, device_type: str = " "):
        if device_type == str("cuda"):
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
        elif device_type == str("mps"):
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
        elif device_type == str("cpu"):
            self.device = torch.device("cpu")
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        print(f"|Device Selected: {self.device}|")