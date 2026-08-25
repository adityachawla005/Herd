import type { Hardware } from '../api'

function Meter({ used, total, label }: { used: number; total: number; label: string }) {
  const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0
  const tone = pct > 85 ? 'bg-rose-500' : pct > 60 ? 'bg-amber-400' : 'bg-emerald-400'
  return (
    <div>
      <div className="flex justify-between text-xs mb-1.5">
        <span className="text-zinc-400">{label}</span>
        <span className="font-mono text-zinc-300">
          {(total - used).toFixed(1)} / {total.toFixed(1)} GB free
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-zinc-800 overflow-hidden">
        <div className={`h-full ${tone} transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-widest text-zinc-500">{label}</div>
      <div className="text-lg font-semibold text-zinc-100 mt-0.5 truncate" title={value}>{value}</div>
      {sub && <div className="text-xs text-zinc-500 font-mono">{sub}</div>}
    </div>
  )
}

export function HardwareCard({ hw }: { hw: Hardware }) {
  const gpu = hw.gpus[0]
  return (
    <section className="rounded-2xl border border-edge bg-panel/70 backdrop-blur p-6">
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="GPU"
          value={gpu ? gpu.name : 'None — CPU only'}
          sub={gpu ? `${gpu.backend} · ${gpu.bandwidth_gbs.toFixed(0)} GB/s${gpu.bandwidth_known ? '' : ' (est)'}` : 'add a GPU for real speed'}
        />
        <Stat label="CPU" value={`${hw.cpu_cores} cores`} sub={`${hw.cpu_threads} threads`} />
        <Stat label="VRAM" value={`${hw.vram_total_gb.toFixed(1)} GB`} sub={`${hw.vram_available_gb.toFixed(1)} GB free`} />
        <Stat label="RAM" value={`${hw.ram_total_gb.toFixed(1)} GB`} sub={`${hw.ram_available_gb.toFixed(1)} GB free`} />
      </div>
      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        {gpu && <Meter label="VRAM in use" used={hw.vram_total_gb - hw.vram_available_gb} total={hw.vram_total_gb} />}
        <Meter label="RAM in use" used={hw.ram_total_gb - hw.ram_available_gb} total={hw.ram_total_gb} />
      </div>
      <div className="mt-5 text-xs text-zinc-500 font-mono truncate">{hw.cpu_name} · {hw.os}</div>
      {hw.notes.map((n) => (
        <p key={n} className="mt-3 text-sm text-amber-300/80 border-l-2 border-amber-500/40 pl-3">{n}</p>
      ))}
    </section>
  )
}
