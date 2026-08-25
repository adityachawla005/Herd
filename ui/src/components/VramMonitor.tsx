import { useEffect, useRef, useState } from 'react'

type Loaded = { tag: string; size_gb: number; idle_s: number; pinned: boolean; holders: string[] }
type Recent = {
  t: number; destination: string; agent?: string; model?: string
  tokens_per_sec?: number | null; cache_hit?: boolean; cost_usd?: number
}
type Status = {
  vram: { vram_total_gb: number; budget_gb: number; used_gb: number; free_gb: number; utilization: number; loaded: Loaded[] }
  agents: Record<string, { tag: string | null; why: string; pinned: boolean; resident: boolean; est_tok_s: number | null }>
  totals: { local_tasks?: number; cloud_tasks?: number; estimated_saved_usd?: number; cloud_spend_usd?: number; recent?: Recent[] }
  calibrated: boolean
  error?: string
}

const SLOT_COLORS = ['bg-emerald-400', 'bg-sky-400', 'bg-violet-400', 'bg-amber-400', 'bg-rose-400']

// A rolling window of utilization samples, so pressure is visible as a shape not a number.
function useHistory(value: number | undefined, size = 60) {
  const [history, setHistory] = useState<number[]>([])
  const last = useRef<number | undefined>(undefined)
  useEffect(() => {
    if (value === undefined || value === last.current) return
    last.current = value
    setHistory((h) => [...h, value].slice(-size))
  }, [value, size])
  return history
}

function Sparkline({ points }: { points: number[] }) {
  if (points.length < 2) return <div className="h-10" />
  const d = points
    .map((p, i) => `${(i / (points.length - 1)) * 100},${34 - Math.min(1, p) * 32}`)
    .join(' L ')
  return (
    <svg viewBox="0 0 100 36" preserveAspectRatio="none" className="w-full h-10">
      <polyline points="" />
      <path d={`M ${d}`} fill="none" stroke="currentColor" strokeWidth="1.2"
            className="text-emerald-400" vectorEffect="non-scaling-stroke" />
      <path d={`M ${d} L 100,36 L 0,36 Z`} className="fill-emerald-400/10" />
    </svg>
  )
}

export function VramMonitor() {
  const [status, setStatus] = useState<Status | null>(null)
  const [connected, setConnected] = useState(false)
  const history = useHistory(status?.vram?.utilization)

  useEffect(() => {
    let ws: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout>
    let closed = false

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/ws/vram`)
      ws.onopen = () => setConnected(true)
      ws.onmessage = (e) => setStatus(JSON.parse(e.data))
      ws.onclose = () => {
        setConnected(false)
        if (!closed) retry = setTimeout(connect, 3000)
      }
      ws.onerror = () => ws?.close()
    }
    connect()
    return () => { closed = true; clearTimeout(retry); ws?.close() }
  }, [])

  if (status?.error) {
    return (
      <section className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-6">
        <div className="font-medium text-amber-300">Scheduler unavailable</div>
        <p className="text-sm text-amber-200/70 mt-1 font-mono">{status.error}</p>
      </section>
    )
  }
  if (!status) {
    return (
      <section className="rounded-2xl border border-edge bg-panel/70 p-10 text-center text-zinc-500 animate-pulse">
        Connecting to the scheduler…
      </section>
    )
  }

  const v = status.vram
  const recent = status.totals?.recent ?? []

  return (
    <section className="rounded-2xl border border-edge bg-panel/70 backdrop-blur p-6">
      <div className="flex items-baseline justify-between flex-wrap gap-2">
        <h2 className="text-sm font-semibold tracking-wide uppercase text-emerald-400">VRAM monitor</h2>
        <span className="flex items-center gap-2 text-xs text-zinc-500">
          <span className={`size-1.5 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
          {connected ? 'live · 2s' : 'reconnecting…'}
        </span>
      </div>

      {/* Stacked bar: one segment per resident model, remainder free. */}
      <div className="mt-5 flex h-8 rounded-lg overflow-hidden bg-zinc-800/70">
        {v.loaded.map((m, i) => (
          <div
            key={m.tag}
            title={`${m.tag} — ${m.size_gb.toFixed(2)}GB`}
            className={`${SLOT_COLORS[i % SLOT_COLORS.length]} transition-all duration-700 flex items-center px-2 overflow-hidden`}
            style={{ width: `${(m.size_gb / v.budget_gb) * 100}%` }}
          >
            <span className="text-[10px] font-mono text-black/70 truncate">{m.tag}</span>
          </div>
        ))}
      </div>
      <div className="mt-2 flex justify-between text-xs font-mono text-zinc-500">
        <span className="text-zinc-300">{v.used_gb.toFixed(2)} GB used</span>
        <span>{v.free_gb.toFixed(2)} GB free of {v.budget_gb.toFixed(1)} GB budget</span>
      </div>

      <div className="mt-4 -mx-1"><Sparkline points={history} /></div>

      <div className="mt-4 space-y-1.5">
        {!v.loaded.length && <div className="text-sm text-zinc-500">Nothing resident.</div>}
        {v.loaded.map((m, i) => (
          <div key={m.tag} className="flex items-center gap-2 text-sm">
            <span className={`size-1.5 rounded-full ${SLOT_COLORS[i % SLOT_COLORS.length]}`} />
            <span className="text-zinc-200 font-mono text-xs">{m.tag}</span>
            {m.pinned && <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-300">pinned</span>}
            <span className="ml-auto text-xs text-zinc-500 font-mono">
              {m.holders.length ? `in use · ${m.holders.join(', ')}` : `idle ${m.idle_s.toFixed(0)}s`}
            </span>
            <span className="text-xs text-zinc-300 font-mono tabular-nums w-16 text-right">{m.size_gb.toFixed(2)} GB</span>
          </div>
        ))}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <div className="text-[11px] uppercase tracking-widest text-zinc-500 mb-2">Agents</div>
          <div className="space-y-1.5">
            {Object.entries(status.agents).map(([name, a]) => (
              <div key={name} className="flex items-center gap-2 text-sm">
                <span className={`size-1.5 rounded-full ${a.resident ? 'bg-emerald-400' : 'bg-zinc-700'}`} />
                <span className="text-zinc-300 w-24">{name}</span>
                <span className={`font-mono text-xs truncate ${
                  !a.tag ? 'text-rose-400' : a.tag.startsWith('cloud/') ? 'text-amber-400' : 'text-zinc-400'
                }`}>{a.tag ?? 'unavailable'}</span>
                {a.est_tok_s != null && (
                  <span className="ml-auto text-xs text-zinc-500 font-mono shrink-0">~{a.est_tok_s.toFixed(0)} tok/s</span>
                )}
              </div>
            ))}
          </div>
        </div>

        <div>
          <div className="text-[11px] uppercase tracking-widest text-zinc-500 mb-2">Recent routing</div>
          {!recent.length && <div className="text-sm text-zinc-600">No tasks yet — try <code className="font-mono text-xs">herd agent "…"</code></div>}
          <div className="space-y-1.5 max-h-44 overflow-y-auto pr-1">
            {recent.map((r, i) => (
              <div key={`${r.t}-${i}`} className="flex items-center gap-2 text-xs">
                <span className={`px-1.5 py-0.5 rounded font-mono ${
                  r.destination === 'local' ? 'bg-emerald-500/15 text-emerald-300' : 'bg-sky-500/15 text-sky-300'
                }`}>{r.destination}</span>
                <span className="text-zinc-400">{r.agent}</span>
                <span className="text-zinc-600 font-mono truncate">{r.model}</span>
                <span className="ml-auto text-zinc-500 font-mono shrink-0">
                  {r.tokens_per_sec ? `${r.tokens_per_sec.toFixed(0)} tok/s` : r.cost_usd ? `$${r.cost_usd}` : ''}
                  {r.cache_hit ? ' · hit' : r.cache_hit === false ? ' · cold' : ''}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-edge flex items-baseline gap-4 flex-wrap text-sm">
        <span className="text-zinc-400">
          <span className="text-emerald-400 font-semibold tabular-nums">{status.totals?.local_tasks ?? 0}</span> local
        </span>
        <span className="text-zinc-400">
          <span className="text-sky-400 font-semibold tabular-nums">{status.totals?.cloud_tasks ?? 0}</span> cloud
        </span>
        <span className="ml-auto text-zinc-400">
          ~<span className="text-emerald-400 font-semibold tabular-nums">${(status.totals?.estimated_saved_usd ?? 0).toFixed(2)}</span> saved
        </span>
      </div>
    </section>
  )
}
