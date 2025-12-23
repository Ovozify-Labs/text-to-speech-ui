from typing import Optional, Dict
from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: Optional[str] = None
    speed: Optional[float] = 0.9
    max_chars: Optional[int] = 200
    cross_fade_duration: Optional[float] = 0.15
