"""Hardware detection. Every probe fails soft: we record a note and keep going."""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict

import psutil

from . import registry

GB = 1024 ** 3


@dataclass
class GPU:
    name: str
    vendor: str            # nvidia | amd | apple | none
    backend: str           # cuda | rocm | metal | cpu
    vram_total_gb: float
    vram_available_gb: float
    bandwidth_gbs: float
    bandwidth_known: bool = True
    index: int = 0

    @property
    def unified(self) -> bool:
        return self.vendor == "apple"


@dataclass
class Hardware:
    gpus: list[GPU] = field(default_factory=list)
    driver: dict = field(default_factory=dict)
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_name: str = "unknown"
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    os: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def gpu(self) -> GPU | None:
        return self.gpus[0] if self.gpus else None

    @property
    def vram_total_gb(self) -> float:
        return sum(g.vram_total_gb for g in self.gpus)

    @property
    def vram_available_gb(self) -> float:
        return sum(g.vram_available_gb for g in self.gpus)

    @property
    def bandwidth_gbs(self) -> float:
        if self.gpu:
            return self.gpu.bandwidth_gbs
        return registry.gpus()["cpu_ram_bandwidth_gbs"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vram_total_gb"] = round(self.vram_total_gb, 2)
        d["vram_available_gb"] = round(self.vram_available_gb, 2)
        d["bandwidth_gbs"] = self.bandwidth_gbs
        return d


def _run(cmd: list[str], timeout: int = 8) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except (subprocess.SubprocessError, OSError):
        return None


# --- NVIDIA ---------------------------------------------------------------

def _nvidia(notes: list[str]) -> list[GPU]:
    gpus = _nvidia_pynvml(notes)
    if gpus:
        return gpus
    return _nvidia_smi(notes)


def _nvidia_pynvml(notes: list[str]) -> list[GPU]:
    try:
        import pynvml
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
    except Exception as e:                                    # driver missing / no GPU
        notes.append(f"NVML unavailable ({_short(e)}); falling back to nvidia-smi.")
        return []
    out = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            bw, known = registry.bandwidth_for(name, "nvidia")
            out.append(GPU(name, "nvidia", "cuda", mem.total / GB, mem.free / GB, bw, known, i))
    except Exception as e:
        notes.append(f"NVML query failed partway ({_short(e)}).")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out


def _nvidia_smi(notes: list[str]) -> list[GPU]:
    out = _run(["nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = []
    for i, line in enumerate(l for l in out.strip().splitlines() if l.strip()):
        try:
            name, total, free = [p.strip() for p in line.split(",")]
            bw, known = registry.bandwidth_for(name, "nvidia")
            gpus.append(GPU(name, "nvidia", "cuda", float(total) / 1024, float(free) / 1024, bw, known, i))
        except ValueError:
            notes.append(f"Could not parse nvidia-smi line: {line!r}")
    return gpus


# --- Apple Silicon --------------------------------------------------------

def _apple(notes: list[str], ram_total_gb: float, ram_avail_gb: float) -> list[GPU]:
    out = _run(["system_profiler", "SPDisplaysDataType"], timeout=25)
    chip = None
    if out:
        m = re.search(r"Chipset Model:\s*(.+)", out)
        if m:
            chip = m.group(1).strip()
    if not chip:
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
        chip = brand.strip() or "Apple Silicon"
        notes.append("system_profiler gave no chipset; used CPU brand string instead.")

    # Unified memory: the GPU may address most of RAM. Ask the kernel for the real cap.
    limit = _run(["sysctl", "-n", "iogpu.wired_limit_mb"])
    if limit and limit.strip().isdigit() and int(limit.strip()) > 0:
        vram_total = int(limit.strip()) / 1024
    else:
        # macOS default: ~75% of RAM below 36GB, ~85% above.
        vram_total = ram_total_gb * (0.85 if ram_total_gb > 36 else 0.75)
        notes.append("Unified memory: GPU budget estimated at the macOS default "
                     f"({vram_total:.0f}GB). Raise it with "
                     "`sudo sysctl iogpu.wired_limit_mb=<mb>`.")
    bw, known = registry.bandwidth_for(chip, "apple")
    if not known:
        notes.append(f"No bandwidth entry for {chip!r} in gpus.json; using a generic Apple figure.")
    return [GPU(chip, "apple", "metal", vram_total, min(vram_total, ram_avail_gb), bw, known, 0)]


# --- AMD ------------------------------------------------------------------

def _amd(notes: list[str]) -> list[GPU]:
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if not out:
        return []
    gpus = []
    for i, line in enumerate(out.strip().splitlines()[1:]):
        nums = re.findall(r"\d+", line)
        if len(nums) < 3:
            continue
        total, used = float(nums[-2]) / GB, float(nums[-1]) / GB
        name = (_run(["rocm-smi", "--showproductname"]) or "AMD GPU").strip().splitlines()[-1][:60]
        bw, known = registry.bandwidth_for(name, "amd")
        gpus.append(GPU(name, "amd", "rocm", total, max(total - used, 0), bw, known, i))
    if gpus:
        notes.append("AMD detected via rocm-smi. Bandwidth and available-VRAM figures are "
                     "coarser than NVIDIA's; treat estimates as ±20%.")
    return gpus


def _short(e: Exception) -> str:
    return str(e).strip().splitlines()[0][:90] or type(e).__name__


def _driver_health(notes: list[str]) -> dict:
    """Is the NVIDIA driver actually able to report on itself?

    A card that holds memory but reports ERR! power, unreadable clocks and an empty
    process table is in a degraded state. Inference may still run, but every timing
    estimate built on its numbers is guesswork — better to say so than to print a
    confident tok/s.
    """
    out = _run(["nvidia-smi", "--query-gpu=power.draw,clocks.sm,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"])
    if not out:
        return {}
    fields = [f.strip() for f in out.strip().splitlines()[0].split(",")]
    labels = ["power draw", "SM clock", "utilization", "memory used"]
    bad = [lab for lab, val in zip(labels, fields) if not val.replace(".", "", 1).isdigit()]
    health = {"degraded": bool(bad), "unreadable": bad}
    if bad:
        notes.append(
            "The NVIDIA driver is in a degraded state: " + ", ".join(bad) +
            " cannot be read (nvidia-smi returns ERR!/not-in-ready-state). Inference may "
            "still work, but speed estimates and the `herd bench` calibration are "
            "unreliable until it recovers. Usually a stuck power state on a hybrid-graphics "
            "laptop — try `sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia` "
            "with nothing using the GPU, or a reboot.")
    return health


def _cpu_name() -> str:
    if platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif platform.system() == "Darwin":
        return (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip() or platform.processor()
    return platform.processor() or platform.machine()


def detect() -> Hardware:
    notes: list[str] = []
    vm = psutil.virtual_memory()
    hw = Hardware(
        cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        cpu_threads=psutil.cpu_count() or 1,
        cpu_name=_cpu_name(),
        ram_total_gb=vm.total / GB,
        ram_available_gb=vm.available / GB,
        os=f"{platform.system()} {platform.release()}",
    )

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        hw.gpus = _apple(notes, hw.ram_total_gb, hw.ram_available_gb)
    else:
        hw.gpus = _nvidia(notes) or _amd(notes)

    if not hw.gpus:
        notes.append("No GPU detected — profiling for CPU inference. "
                     "If you do have a GPU: NVIDIA needs a working driver (`nvidia-smi`), "
                     "AMD needs ROCm (`rocm-smi`).")
        bw = registry.gpus()["cpu_ram_bandwidth_gbs"]
        notes.append(f"CPU RAM bandwidth assumed {bw} GB/s (dual-channel DDR4). "
                     "Edit cpu_ram_bandwidth_gbs in gpus.json if yours differs.")
    else:
        for g in hw.gpus:
            if not g.bandwidth_known:
                notes.append(f"{g.name!r} is not in gpus.json — using a generic "
                             f"{g.vendor} bandwidth of {g.bandwidth_gbs:.0f} GB/s. "
                             "Add the real figure for accurate tok/s.")
            if g.vram_total_gb <= 0:
                notes.append(f"{g.name!r} reported 0GB VRAM; its numbers are unusable.")
        if hw.gpus[0].vendor == "nvidia":
            hw.driver = _driver_health(notes)
        if len(hw.gpus) > 1:
            notes.append(f"{len(hw.gpus)} GPUs found. Fit math assumes tensor-parallel "
                         "across all of them; single-GPU runs get the first card only.")

    hw.notes = notes
    return hw
