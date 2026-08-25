export type GPU = {
  name: string; vendor: string; backend: string
  vram_total_gb: number; vram_available_gb: number
  bandwidth_gbs: number; bandwidth_known: boolean
}

export type Hardware = {
  gpus: GPU[]; cpu_cores: number; cpu_threads: number; cpu_name: string
  ram_total_gb: number; ram_available_gb: number; os: string; notes: string[]
  vram_total_gb: number; vram_available_gb: number; bandwidth_gbs: number
}

export type Fit = {
  model: string; quant: string; params_b: number; arch: string; context: number
  weight_gb: number; kv_gb: number; overhead_gb: number; total_gb: number; active_gb: number
  fits_vram: boolean; fits_offload: boolean; tokens_per_sec: number
  gpu_layers: number; total_layers: number; tier: 'great' | 'offload' | 'no'
  quality: number; use_cases: string[]; ollama: string; notes: string[]
  backend: string; model_ref: string
}

export type Tip = {
  id: string; title: string; detail: string; impact: string; command: string; tags: string[]
}

export type Hybrid = {
  local: { use_case: string; share: number; model: string; quant: string; tokens_per_sec: number }[]
  cloud: string[]; local_coverage: number
  savings_per_1000_calls_usd: number; cloud_cost_per_1000_calls_usd: number
  router: string | null; workhorse: string | null
}

export type BackendInfo = {
  id: string; label: string; available: boolean; detail: string
  can_evict: boolean; owns_card: boolean; reports_vram: boolean
  quants: string[]; runtime_overhead_gb: number; install: string; notes: string; models: string[]
}

export type Recommendation = {
  hardware: Hardware; context: number; kv_quant: string
  backend: string; backends: BackendInfo[]
  runs_great: Fit[]; runs_offload: Fit[]; wont_fit: Fit[]
  hybrid: Hybrid; optimizations: Tip[]; notes: string[]
}

export async function getRecommendation(
  context: number, kvQuant: string, backend?: string,
): Promise<Recommendation> {
  const q = new URLSearchParams({ context: String(context), kv_quant: kvQuant, limit: '12' })
  if (backend) q.set('backend', backend)
  const r = await fetch(`/api/recommend?${q}`)
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `API returned ${r.status}`)
  return r.json()
}
