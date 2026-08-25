import { useState } from 'react'
import type { Tip } from '../api'

const TAG_TONE: Record<string, string> = {
  memory: 'bg-emerald-500/15 text-emerald-300',
  speed: 'bg-sky-500/15 text-sky-300',
  quality: 'bg-violet-500/15 text-violet-300',
  runtime: 'bg-amber-500/15 text-amber-300',
  quant: 'bg-fuchsia-500/15 text-fuchsia-300',
}

function Copy({ text }: { text: string }) {
  const [done, setDone] = useState(false)
  return (
    <button
      onClick={() => { navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1200) }}
      className="shrink-0 text-[11px] px-2 py-1 rounded-md border border-edge text-zinc-400 hover:text-zinc-100 hover:border-zinc-600 transition"
    >
      {done ? 'copied' : 'copy'}
    </button>
  )
}

function TipCard({ tip }: { tip: Tip }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded-xl border border-edge bg-zinc-900/40 overflow-hidden">
      <button onClick={() => setOpen(!open)} className="w-full text-left px-4 py-3 flex items-start gap-3 hover:bg-white/[0.03] transition">
        <span className={`mt-1 shrink-0 text-zinc-600 transition-transform ${open ? 'rotate-90' : ''}`}>›</span>
        <div className="min-w-0 flex-1">
          <div className="font-medium text-zinc-100">{tip.title}</div>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            {tip.impact && (
              <span className="text-[11px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 font-mono">{tip.impact}</span>
            )}
            {tip.tags.map((t) => (
              <span key={t} className={`text-[10px] px-1.5 py-0.5 rounded ${TAG_TONE[t] ?? 'bg-zinc-800 text-zinc-400'}`}>{t}</span>
            ))}
          </div>
        </div>
      </button>
      {open && (
        <div className="px-4 pb-4 pl-11 space-y-3">
          <p className="text-sm text-zinc-400 leading-relaxed">{tip.detail}</p>
          {tip.command && (
            <div className="flex items-center gap-2 rounded-lg bg-black/50 border border-edge px-3 py-2">
              <code className="font-mono text-xs text-emerald-300 overflow-x-auto flex-1 whitespace-pre">{tip.command}</code>
              <Copy text={tip.command} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function Optimizations({ tips }: { tips: Tip[] }) {
  if (!tips.length) return null
  return (
    <section className="rounded-2xl border border-edge bg-panel/70 backdrop-blur p-6">
      <div className="flex items-baseline justify-between flex-wrap gap-2">
        <h2 className="text-sm font-semibold tracking-wide uppercase text-fuchsia-400">Optimizations</h2>
        <span className="text-xs text-zinc-500">tuned to this machine — click any row for the command</span>
      </div>
      <div className="mt-5 grid gap-2 lg:grid-cols-2">
        {tips.map((t) => <TipCard key={t.id} tip={t} />)}
      </div>
    </section>
  )
}
