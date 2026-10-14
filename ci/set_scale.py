"""Set the primary display's Windows scale (100/125/150/...) live, without a sign-out.

Uses DisplayConfigSetDeviceInfo with the DPI-scale request types Settings > Display uses
(-3 get, -4 set; the same calls as github.com/imniko/SetDPI). Prints the monitor DPI after.
"""

import ctypes
import sys
from ctypes import wintypes

DPI_VALUES = [100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500]
QDC_ONLY_ACTIVE_PATHS = 2


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class HEADER(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int32), ("size", ctypes.c_uint32), ("adapterId", LUID), ("id", ctypes.c_uint32)]


class GET(ctypes.Structure):
    _fields_ = [("header", HEADER), ("minScaleRel", ctypes.c_int32), ("curScaleRel", ctypes.c_int32),
                ("maxScaleRel", ctypes.c_int32)]


class SET(ctypes.Structure):
    _fields_ = [("header", HEADER), ("scaleRel", ctypes.c_int32)]


class PATH(ctypes.Structure):
    _fields_ = [("srcAdapter", LUID), ("srcId", ctypes.c_uint32), ("srcModeIdx", ctypes.c_uint32),
                ("srcFlags", ctypes.c_uint32), ("rest", ctypes.c_byte * 52)]


class MODE(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_byte * 64)]


def monitor_dpi():
    shcore = ctypes.windll.shcore
    user32 = ctypes.windll.user32
    mon = user32.MonitorFromPoint(wintypes.POINT(0, 0), 1)
    x, y = ctypes.c_uint(), ctypes.c_uint()
    shcore.GetDpiForMonitor(mon, 0, ctypes.byref(x), ctypes.byref(y))
    return x.value


def main():
    want = int(sys.argv[1])
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    user32 = ctypes.windll.user32
    np, nm = ctypes.c_uint32(), ctypes.c_uint32()
    assert user32.GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, ctypes.byref(np), ctypes.byref(nm)) == 0
    paths, modes = (PATH * np.value)(), (MODE * nm.value)()
    assert ctypes.sizeof(PATH) == 72 and ctypes.sizeof(MODE) == 64
    assert user32.QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, ctypes.byref(np), paths, ctypes.byref(nm), modes, None) == 0
    print("monitor DPI before:", monitor_dpi())
    for p in paths[: np.value]:
        g = GET()
        g.header.type, g.header.size, g.header.adapterId, g.header.id = -3, ctypes.sizeof(GET), p.srcAdapter, p.srcId
        rc = user32.DisplayConfigGetDeviceInfo(ctypes.byref(g))
        rec = abs(g.minScaleRel)
        cur = DPI_VALUES[rec + g.curScaleRel] if 0 <= rec + g.curScaleRel < len(DPI_VALUES) else None
        print(f"source {p.srcId}: get rc={rc} min={g.minScaleRel} cur={g.curScaleRel} max={g.maxScaleRel} "
              f"recommended={DPI_VALUES[rec]} current={cur}")
        s = SET()
        s.header.type, s.header.size, s.header.adapterId, s.header.id = -4, ctypes.sizeof(SET), p.srcAdapter, p.srcId
        s.scaleRel = DPI_VALUES.index(want) - rec
        rc = user32.DisplayConfigSetDeviceInfo(ctypes.byref(s))
        print(f"set {want}% rc={rc}")
    print("monitor DPI after:", monitor_dpi(), "->", round(monitor_dpi() / 96 * 100), "%")


if __name__ == "__main__":
    main()
