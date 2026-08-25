import type { Hybrid } from '../api'

export function HybridPlan({ h }: { h: Hybrid }) {
  const pct = Math.round(h.local_coverage * 100)
  return (
    <section className="rounded-2xl border border-edge bg-panel/70 backdrop-blur p-6">
      <div className="flex items-baseline justify-between flex-wrap gap-2">
        <h2 className="text-sm font-semibold tracking-wide uppercase text-sky-400">Hybrid plan</h2>
        <span className="text-xs text-zinc-500">what stays local, what goes to the cloud</span>
      </div>

      <div className="mt-5 flex h-2.5 rounded-full overflow-hidden bg-zinc-800">
        <div className="bg-emerald-400 transition-all duration-500" style={{ width: `${pct}%` }} />
        <div className="bg-sky-500/60 transition-all duration-500" style={{ width: `${100 - pct}%` }} />
      </div>
      <div className="mt-2 flex justify-between text-xs">
        <span className="text-emerald-400 font-medium">{pct}% local</span>
        <span className="text-sky-400 font-medium">{100 - pct}% cloud</span>
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <div>
          <div className="text-[11px] uppercase tracking-widest text-zinc-500 mb-2">Run locally</div>
          <ul className="space-y-2">
            {h.local.map((a) => (
              <li key={a.use_case} className="flex items-center gap-2 text-sm">
                <span className="size-1.5 rounded-full bg-emerald-400" />
                <span className="text-zinc-400 w-20 shrink-0">{a.use_case}</span>
                <span className="text-zinc-100 font-medium truncate">{a.model}</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">{a.quant}</span>
                <span className="ml-auto font-mono text-xs text-emerald-400/80 tabular-nums shrink-0">
                  ~{a.tokens_per_sec.toFixed(0)} tok/s
                </span>
              </li>
            ))}
            {!h.local.length && <li className="text-sm text-zinc-500">Nothing runs well enough locally yet.</li>}
          </ul>
        </div>
        <div>
          <div className="text-[11px] uppercase tracking-widest text-zinc-500 mb-2">Route to cloud</div>
          <ul className="space-y-2">
            {h.cloud.map((c) => (
              <li key={c} className="flex items-center gap-2 text-sm">
                <span className="size-1.5 rounded-full bg-sky-500" />
                <span className="text-zinc-300">{c}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-edge flex items-baseline gap-2 flex-wrap">
        <span className="text-2xl font-semibold text-emerald-400 tabular-nums">
          ${h.savings_per_1000_calls_usd.toFixed(2)}
        </span>
        <span className="text-sm text-zinc-400">
          saved per 1000 calls — of ${h.cloud_cost_per_1000_calls_usd.toFixed(2)} if every call went to a frontier API
        </span>
      </div>
    </section>
  )
}
