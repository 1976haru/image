from dataclasses import asdict, dataclass


@dataclass
class ChannelPreset:
    name: str
    text_safe_ratio_16x9: float = 0.35
    person_anchor_16x9: str = "right"
    person_ratio_16x9: float = 0.42
    person_anchor_9x16: str = "center"
    person_ratio_9x16: float = 0.72
    enhance_strength: float = 1.0
    description: str = ""

PRESETS: dict[str, ChannelPreset] = {
    "OldPopLounge": ChannelPreset(
        name="OldPopLounge",
        text_safe_ratio_16x9=0.40,
        person_anchor_16x9="right",
        person_ratio_16x9=0.38,
        person_anchor_9x16="center",
        person_ratio_9x16=0.74,
        enhance_strength=0.88,
        description="좌측 40% 텍스트 공간 · 인물 우측 35~40% · 차분한 플레이리스트용"
    ),
    "Tokyo ChillRap": ChannelPreset(
        name="Tokyo ChillRap",
        text_safe_ratio_16x9=0.34,
        person_anchor_16x9="right",
        person_ratio_16x9=0.46,
        person_anchor_9x16="center",
        person_ratio_9x16=0.76,
        enhance_strength=0.94,
        description="카페/도시 배경 유지 · 인물 40~50% · 일본 채널용"
    ),
    "Generic Playlist": ChannelPreset(
        name="Generic Playlist",
        text_safe_ratio_16x9=0.30,
        person_anchor_16x9="right",
        person_ratio_16x9=0.45,
        person_anchor_9x16="center",
        person_ratio_9x16=0.72,
        enhance_strength=1.0,
        description="범용 플레이리스트 프리셋"
    ),
}

def preset_to_dict(preset: ChannelPreset):
    return asdict(preset)
