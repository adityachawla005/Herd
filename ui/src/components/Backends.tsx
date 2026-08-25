import type { BackendInfo } from '../api'

/** What can actually run models here, and what the scheduler can do with each. */
export function Backends({ backends, active }: { backends: BackendInfo[]; active: string }) {
  return (
    <section className="rounded-2xl border border-edge bg-panel/70 backdrop-blur p-6">
      <div className="flex items-baseline justify-between flex-wrap gap-2">
        <h2 className="text-sm font-semibold tracking-wide uppercase text-amber-400">Backends</h2>
        <span className="text-xs text-zinc-500">sizing above is for the selected one</span>
      </div>
      <div className="mt-4 grid gap-2 sm:grid-cols-2">
        {backends.map((b) => (
          <div
            key={b.id}
            className={`rounded-xl border p-3 transition ${
              b.id === active ? 'border-emerald-500/40 bg-emerald-500/5' : 'border-edge bg-zinc-900/40'
            }`}
          >
            <div className="flex items-center gap-2">
              <span className={`size-1.5 rounded-full ${b.available ? 'bg-emerald-400' : 'bg-zinc-700'}`} />
              <span className="font-medium text-zinc-100">{b.label}</span>
              {b.id === active && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300">in use</span>
              )}
              <span className="ml-auto text-[10px] font-mono text-zinc-500">{b.quants.join('/')}</span>
            </div>
            <p className="mt-1.5 text-xs text-zinc-500 leading-relaxed">{b.detail}</p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                b.can_evict ? 'bg-zinc-800 text-zinc-400' : 'bg-amber-500/15 text-amber-300'
              }`}>{b.can_evict ? 'schedulable' : 'cannot evict'}</span>
              {b.owns_card && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300">reserves the GPU</span>
              )}
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">
                {b.runtime_overhead_gb.toFixed(1)}GB runtime
              </span>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
