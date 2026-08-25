import { useEffect, useState } from 'react'
import { getRecommendation, type Recommendation } from './api'
import { HardwareCard } from './components/HardwareCard'
import { ModelTable } from './components/ModelTable'
import { HybridPlan } from './components/HybridPlan'
import { Optimizations } from './components/Optimizations'
import { VramMonitor } from './components/VramMonitor'
import { Backends } from './components/Backends'

const CONTEXTS = [2048, 4096, 8192, 16384, 32768]
const KV_QUANTS = [
  { id: 'fp16', label: 'fp16' },
  { id: 'q8', label: 'q8' },
  { id: 'q4', label: 'q4' },
]

function Toggle<T extends string | number>({ options, value, onChange, label }: {
  options: { id: T; label: string }[]; value: T; onChange: (v: T) => void; label: string
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] uppercase tracking-widest text-zinc-500">{label}</span>
      <div className="flex rounded-lg border border-edge bg-zinc-900/60 p-0.5">
        {options.map((o) => (
          <button
            key={o.id}
            onClick={() => onChange(o.id)}
            className={`px-2.5 py-1 text-xs font-mono rounded-md transition ${
              o.id === value ? 'bg-zinc-700 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const [data, setData] = useState<Recommendation | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [context, setContext] = useState(4096)
  const [kvQuant, setKvQuant] = useState('fp16')
  const [tab, setTab] = useState<'profile' | 'scheduler'>('profile')
  const [backend, setBackend] = useState<string>('')

  useEffect(() => {
    setLoading(true)
    getRecommendation(context, kvQuant, backend)
      .then((d) => { setData(d); setError(null); if (!backend) setBackend(d.backend) })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [context, kvQuant, backend])

  return (
    <div className="min-h-screen text-zinc-200 font-sans">
      <div className="mx-auto max-w-6xl px-6 py-12 space-y-6">
        <header className="flex items-end justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-zinc-50">
              Herd<span className="text-emerald-400">.</span>
            </h1>
            <p className="text-sm text-zinc-500 mt-1">
              What this machine can actually run — and how to make it run better.
            </p>
          </div>
          <div className="flex items-center gap-5 flex-wrap">
            <Toggle
              label="view"
              value={tab}
              onChange={setTab}
              options={[{ id: 'profile' as const, label: 'profile' }, { id: 'scheduler' as const, label: 'scheduler' }]}
            />
            {tab === 'profile' && (
              <>
                {data && data.backends.length > 1 && (
                  <Toggle
                    label="backend"
                    value={backend || data.backend}
                    onChange={setBackend}
                    options={data.backends.map((b) => ({
                      id: b.id,
                      label: b.available ? b.label.split(' ')[0].toLowerCase() : `${b.id}*`,
                    }))}
                  />
                )}
                <Toggle
                  label="context"
                  value={context}
                  onChange={setContext}
                  options={CONTEXTS.map((c) => ({ id: c, label: c >= 1024 ? `${c / 1024}k` : `${c}` }))}
                />
                <Toggle label="kv cache" value={kvQuant} onChange={setKvQuant} options={KV_QUANTS} />
              </>
            )}
          </div>
        </header>

        {tab === 'scheduler' && <VramMonitor />}

        {tab === 'profile' && error && (
          <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-6">
            <div className="font-medium text-rose-300">Could not profile this machine</div>
            <p className="text-sm text-rose-200/70 mt-1 font-mono">{error}</p>
          </div>
        )}

        {tab === 'profile' && data && (
          <div className={`space-y-6 transition-opacity duration-200 ${loading ? 'opacity-50' : ''}`}>
            <HardwareCard hw={data.hardware} />
            <Backends backends={data.backends} active={data.backend} />
            <ModelTable title="Runs great" subtitle="fully resident in VRAM" fits={data.runs_great} />
            <ModelTable title="Runs with offload" subtitle="part of the model streams from RAM" fits={data.runs_offload} />
            <ModelTable title="Won't fit" subtitle="even at 4-bit" fits={data.wont_fit} />
            <HybridPlan h={data.hybrid} />
            <Optimizations tips={data.optimizations} />
          </div>
        )}

        {tab === 'profile' && !data && loading && (
          <div className="rounded-2xl border border-edge bg-panel/70 p-12 text-center text-zinc-500 animate-pulse">
            Profiling hardware…
          </div>
        )}

        <footer className="pt-6 text-xs text-zinc-600 font-mono">
          {tab === 'profile' && data && `${data.backend} · ${data.context} ctx · ${data.kv_quant} kv · ${data.hardware.bandwidth_gbs.toFixed(0)} GB/s`}
          {' · '}estimates, not benchmarks — verify with a real run
        </footer>
      </div>
    </div>
  )
}
