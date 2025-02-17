import matplotlib.colors as mcolors

DEFAULT_COLORS = ["#0A9F9D", "#CEB175", "#E54E21", "#6C8645", "#C18748"]
DEFAULT_COLORS.extend(
    ["#C52E19", "#AC9765", "#54D8B1", "#b67c3b", "#175149", "#AF4E24"]
)
DEFAULT_COLORS.extend(["#FBA72A", "#D3D4D8", "#CB7A5C", "#5785C1"])
DEFAULT_COLORS.extend(["#FF0000", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6"])
DEFAULT_COLORS.extend(["#ECCBAE", "#046C9A", "#D69C4E", "#ABDDDE", "#000000"])

ZISSOU1_CONTINUOUS = [
    "#3A9AB2",
    "#6FB2C1",
    "#91BAB6",
    "#A5C2A3",
    "#BDC881",
    "#DCCB4E",
    "#E3B710",
    "#E79805",
    "#EC7A05",
    "#EF5703",
    "#F11B00",
]


DEFAULT_CMAP = mcolors.LinearSegmentedColormap.from_list("Zissou1Continuous", ZISSOU1_CONTINUOUS)
