"""In-memory fakes for the macOS AppleSMC / IOReport ctypes layer.

These let the real apple.py logic (key discovery, float decode, temperature
averaging, energy->watts conversion, GPU-channel selection, baseline/diff
power windowing, failure latching) run end-to-end on a Linux box, including
the success paths the GitHub-hosted macOS VMs could not reach.

Two seams are used:
  * Temperature: patch apple._load_iokit + _SMCConnection._open + ._call so
    every SMC user-client round trip is served from a dict of fake keys.
  * Power: patch apple._load_cf + _load_ioreport + _cfstr + _from_cfstr so
    IOReport subscription/sampling is served from scripted channel snapshots.
"""

import ctypes
import struct

SUBSCRIBED_SENTINEL = 0xBEEF  # what IOReportCreateSubscription writes back


# --------------------------- AppleSMC (temperature) ---------------------------


def build_smc_keys(float_keys, raw_keys=None):
    """float_keys: {name: value_float} stored as 'flt ' 4-byte LE.
    raw_keys: {name: (type_str, bytes)} for non-float keys (e.g. '#KEY')."""
    keys = {}
    for name, val in float_keys.items():
        keys[name] = ("flt ", struct.pack("<f", val))
    if raw_keys:
        keys.update(raw_keys)
    return keys


def install_fake_smc(apple, monkeypatch, float_keys, raw_keys=None,
                     fail_open=False, bad_result_keys=()):
    """Wire a fake AppleSMC backed by the given keys onto the apple module."""
    keys = build_smc_keys(float_keys, raw_keys)
    # Ordered key list the #KEY count + KEY_AT_INDEX enumeration walk.
    key_list = list(keys.keys())
    # The '#KEY' count must reflect the number of enumerable keys.
    keys.setdefault("#KEY", ("ui32", len(key_list).to_bytes(4, "big")))
    if "#KEY" not in key_list:
        key_list.insert(0, "#KEY")
        keys["#KEY"] = ("ui32", len(key_list).to_bytes(4, "big"))

    A = apple

    def fake_open(self):
        if fail_open:
            raise OSError("simulated AppleSMCKeysEndpoint open failure")
        return 0xC0FFEE  # fake connection handle

    def fake_call(self, ival):
        oval = A._SMCKeyData()
        cmd = ival.data8
        if cmd == A._SMC_CMD_KEY_AT_INDEX:
            idx = ival.data32
            if idx >= len(key_list):
                raise OSError("index out of range")
            oval.key = A._fourcc(key_list[idx])
            oval.result = 0
            return oval

        key = A._fourcc_str(ival.key)
        if key in bad_result_keys:
            raise OSError(f"SMC result code for {key}")
        if key not in keys:
            raise OSError(f"unknown key {key}")
        type_str, data = keys[key]

        if cmd == A._SMC_CMD_KEY_INFO:
            oval.key_info.data_size = len(data)
            oval.key_info.data_type = A._fourcc(type_str)
            oval.result = 0
        elif cmd == A._SMC_CMD_READ_BYTES:
            for i, b in enumerate(data[:32]):
                oval.bytes[i] = b
            oval.result = 0
        else:
            raise OSError(f"unexpected cmd {cmd}")
        return oval

    monkeypatch.setattr(A, "_load_iokit", lambda: object())
    monkeypatch.setattr(A._SMCConnection, "_open", fake_open)
    monkeypatch.setattr(A._SMCConnection, "_call", fake_call)


# --------------------------- IOReport (power) ---------------------------


class FakeCF:
    def __init__(self):
        self.released = []

    def CFDictionaryGetValue(self, delta, key):
        # delta IS the channel-list snapshot in our fakes.
        return delta

    def CFArrayGetCount(self, items):
        return len(items)

    def CFArrayGetValueAtIndex(self, items, i):
        return items[i]

    def CFRelease(self, ref):
        self.released.append(ref)


class FakeIOR:
    """Scripted IOReport. `windows` is a list of channel snapshots; each
    gpu_power_w() poll consumes the next snapshot as the *delta* contents."""

    def __init__(self, windows, channels_handle="ORIG_CHANNELS",
                 subscription_ok=True, samples_ok=True, delta_ok=True):
        self._windows = list(windows)
        self._channels_handle = channels_handle
        self._subscription_ok = subscription_ok
        self._samples_ok = samples_ok
        self._delta_ok = delta_ok
        self._sample_id = 0
        self.sample_channel_args = []  # channels arg passed to each CreateSamples
        self._sample_to_window = {}

    def IOReportCopyChannelsInGroup(self, group, a, b, c, d):
        return self._channels_handle

    def IOReportCreateSubscription(self, allocator, channels, subbed_ptr,
                                   depth, opts):
        if not self._subscription_ok:
            return None
        # Write the subscribed-channels descriptor into the out-param, the way
        # the real API does. apple.py's fix is to use *this* for sampling.
        try:
            subbed_ptr._obj.value = SUBSCRIBED_SENTINEL
        except Exception:
            pass
        return 0x5AB  # subscription handle

    def IOReportCreateSamples(self, sub, channels, opts):
        # Normalize a c_void_p descriptor down to its integer value so tests can
        # compare against SUBSCRIBED_SENTINEL regardless of wrapper type.
        val = channels.value if isinstance(channels, ctypes.c_void_p) else channels
        self.sample_channel_args.append(val)
        if not self._samples_ok:
            return None
        self._sample_id += 1
        sid = self._sample_id
        # Each sample maps to the next scripted window snapshot.
        if self._windows:
            self._sample_to_window[sid] = self._windows.pop(0)
        else:
            self._sample_to_window[sid] = []
        return sid

    def IOReportCreateSamplesDelta(self, prev_sample, cur_sample, opts):
        if not self._delta_ok:
            return None
        # The delta carries the *current* sample's scripted channel snapshot.
        return self._sample_to_window.get(cur_sample, [])

    def IOReportChannelGetChannelName(self, item):
        return item["name"]

    def IOReportChannelGetUnitLabel(self, item):
        return item["unit"]

    def IOReportSimpleGetIntegerValue(self, item, idx):
        return item["energy"]


def install_fake_ioreport(apple, monkeypatch, ior):
    monkeypatch.setattr(apple, "_load_cf", lambda: FakeCF())
    monkeypatch.setattr(apple, "_load_ioreport", lambda: ior)
    # cfstr returns an opaque sentinel; from_cfstr passes our plain strings through.
    monkeypatch.setattr(apple, "_cfstr", lambda cf, s: ("cfstr", s))
    monkeypatch.setattr(apple, "_from_cfstr", lambda cf, ref: ref)


def gpu_channel(name, unit, energy):
    return {"name": name, "unit": unit, "energy": energy}
