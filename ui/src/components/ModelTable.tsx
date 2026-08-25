import type { Fit } from '../api'

const TONES = {
  great: { dot: 'bg-emerald-400', text: 'text-emerald-400', bar: 'bg-emerald-400/80', ring: 'ring-emerald-500/20' },
  offload: { dot: 'bg-amber-400', text: 'text-amber-400', bar: 'bg-amber-400/80', ring: 'ring-amber-500/20' },
  no: { dot: 'bg-rose-500', text: 'text-rose-400', bar: 'bg-rose-500/60', ring: 'ring-rose-500/20' },
} as const

function Row({ fit, maxGb }: { fit: Fit; maxGb: number }) {
  const t = TONES[fit.tier]
  const breakdown = `${fit.weight_gb.toFixed(2)}GB weights + ${fit.kv_gb.toFixed(2)}GB KV cache + ${fit.overhead_gb}GB runtime`
  return (
    <div className="group grid grid-cols-12 items-center gap-3 px-4 py-3 rounded-xl hover:bg-white/[0.03] transition">
      <div className="col-span-12 sm:col-span-4 flex items-center gap-2.5 min-w-0">
        <span className={`size-1.5 rounded-full ${t.dot} shrink-0`} />
        <span className="font-medium text-zinc-100 truncate">{fit.model}</span>
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 shrink-0">{fit.quant}</span>
        {fit.arch === 'moe' && (
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-300 shrink-0">MoE</span>
        )}
      </div>

      <div className="col-span-6 sm:col-span-3" title={breakdown}>
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-sm text-zinc-200 tabular-nums">{fit.total_gb.toFixed(1)} GB</span>
          {fit.tier === 'offload' && (
            <span className="text-[11px] text-amber-400/70 font-mono">{fit.gpu_layers}/{fit.total_layers} on GPU</span>
          )}
        </div>
        <div className="mt-1 h-1 rounded-full bg-zinc-800/80 overflow-hidden">
          <div className={`h-full ${t.bar}`} style={{ width: `${Math.min(100, (fit.total_gb / maxGb) * 100)}%` }} />
        </div>
      </div>

      <div className="col-span-6 sm:col-span-2 text-right sm:text-left">
        {fit.tier === 'no' ? (
          <span className="text-zinc-600 font-mono text-sm">—</span>
        ) : (
          <span className={`font-mono text-sm tabular-nums ${t.text}`}>~{fit.tokens_per_sec.toFixed(0)} tok/s</span>
        )}
      </div>

      <div className="col-span-12 sm:col-span-3 flex flex-wrap gap-1.5">
        {fit.use_cases.map((u) => (
          <span key={u} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800/70 text-zinc-400">{u}</span>
        ))}
      </div>
    </div>
  )
}

export function ModelTable({ title, subtitle, fits }: { title: string; subtitle: string; fits: Fit[] }) {
  if (!fits.length) return null
  const tone = TONES[fits[0].tier]
  const maxGb = Math.max(...fits.map((f) => f.total_gb))
  return (
    <section className={`rounded-2xl border border-edge bg-panel/70 backdrop-blur ring-1 ${tone.ring}`}>
      <header className="px-6 pt-5 pb-3 flex items-baseline gap-3 flex-wrap">
        <h2 className={`text-sm font-semibold tracking-wide uppercase ${tone.text}`}>{title}</h2>
        <span className="text-xs text-zinc-500">{subtitle}</span>
        <span className="ml-auto text-xs text-zinc-600 font-mono">{fits.length}</span>
      </header>
      <div className="px-2 pb-3">
        {fits.map((f) => <Row key={f.model + f.quant} fit={f} maxGb={maxGb} />)}
      </div>
    </section>
  )
}
