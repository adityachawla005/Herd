package main

import (
	"fmt"
	"os"
	"strings"
)

type GPU struct {
	Name            string  `json:"name"`
	Vendor          string  `json:"vendor"`
	Backend         string  `json:"backend"`
	VRAMTotalGB     float64 `json:"vram_total_gb"`
	VRAMAvailableGB float64 `json:"vram_available_gb"`
	BandwidthGBs    float64 `json:"bandwidth_gbs"`
	BandwidthKnown  bool    `json:"bandwidth_known"`
}

type Hardware struct {
	GPUs            []GPU    `json:"gpus"`
	CPUCores        int      `json:"cpu_cores"`
	CPUThreads      int      `json:"cpu_threads"`
	CPUName         string   `json:"cpu_name"`
	RAMTotalGB      float64  `json:"ram_total_gb"`
	RAMAvailableGB  float64  `json:"ram_available_gb"`
	OS              string   `json:"os"`
	Notes           []string `json:"notes"`
	VRAMTotalGB     float64  `json:"vram_total_gb"`
	VRAMAvailableGB float64  `json:"vram_available_gb"`
	BandwidthGBs    float64  `json:"bandwidth_gbs"`
}

type Fit struct {
	Model        string   `json:"model"`
	Quant        string   `json:"quant"`
	ParamsB      float64  `json:"params_b"`
	Arch         string   `json:"arch"`
	Context      int      `json:"context"`
	WeightGB     float64  `json:"weight_gb"`
	KVGB         float64  `json:"kv_gb"`
	OverheadGB   float64  `json:"overhead_gb"`
	TotalGB      float64  `json:"total_gb"`
	ActiveGB     float64  `json:"active_gb"`
	FitsVRAM     bool     `json:"fits_vram"`
	FitsOffload  bool     `json:"fits_offload"`
	TokensPerSec float64  `json:"tokens_per_sec"`
	GPULayers    int      `json:"gpu_layers"`
	TotalLayers  int      `json:"total_layers"`
	Tier         string   `json:"tier"`
	Quality      float64  `json:"quality"`
	UseCases     []string `json:"use_cases"`
	Ollama       string   `json:"ollama"`
	Backend      string   `json:"backend"`
	ModelRef     string   `json:"model_ref"`
	Notes        []string `json:"notes"`
}

type Tip struct {
	ID      string   `json:"id"`
	Title   string   `json:"title"`
	Detail  string   `json:"detail"`
	Impact  string   `json:"impact"`
	Command string   `json:"command"`
	Tags    []string `json:"tags"`
}

type Assignment struct {
	UseCase      string  `json:"use_case"`
	Share        float64 `json:"share"`
	Model        string  `json:"model"`
	Quant        string  `json:"quant"`
	Ollama       string  `json:"ollama"`
	TokensPerSec float64 `json:"tokens_per_sec"`
}

type Hybrid struct {
	Local       []Assignment `json:"local"`
	Cloud       []string     `json:"cloud"`
	Coverage    float64      `json:"local_coverage"`
	Savings1k   float64      `json:"savings_per_1000_calls_usd"`
	CloudCost1k float64      `json:"cloud_cost_per_1000_calls_usd"`
	Router      string       `json:"router"`
	Workhorse   string       `json:"workhorse"`
}

type Calibration struct {
	TokenOverheadMS float64  `json:"token_overhead_ms"`
	Measured        bool     `json:"measured"`
	Unreliable      bool     `json:"unreliable"`
	Reason          string   `json:"reason"`
	EffectiveBW     *float64 `json:"effective_bandwidth_gbs"`
}

type BackendInfo struct {
	ID          string   `json:"id"`
	Label       string   `json:"label"`
	Available   bool     `json:"available"`
	Detail      string   `json:"detail"`
	RefKey      string   `json:"ref_key"`
	Install     string   `json:"install"`
	Notes       string   `json:"notes"`
	CanEvict    bool     `json:"can_evict"`
	OwnsCard    bool     `json:"owns_card"`
	ReportsVRAM bool     `json:"reports_vram"`
	Quants      []string `json:"quants"`
	OverheadGB  float64  `json:"runtime_overhead_gb"`
	Models      []string `json:"models"`
}

type Recommendation struct {
	Hardware      Hardware      `json:"hardware"`
	Backend       string        `json:"backend"`
	Backends      []BackendInfo `json:"backends"`
	Context       int           `json:"context"`
	KVQuant       string        `json:"kv_quant"`
	RunsGreat     []Fit         `json:"runs_great"`
	RunsOffload   []Fit         `json:"runs_offload"`
	WontFit       []Fit         `json:"wont_fit"`
	Hybrid        Hybrid        `json:"hybrid"`
	Optimizations []Tip         `json:"optimizations"`
	Calibration   Calibration   `json:"calibration"`
	Notes         []string      `json:"notes"`
}

type OptimizeResult struct {
	Hardware      Hardware `json:"hardware"`
	Fit           *Fit     `json:"fit"`
	Optimizations []Tip    `json:"optimizations"`
	Notes         []string `json:"notes"`
}

// --- colour ---------------------------------------------------------------

var useColor = true

func initColor(noColor bool) {
	if noColor || os.Getenv("NO_COLOR") != "" || os.Getenv("TERM") == "dumb" {
		useColor = false
		return
	}
	st, err := os.Stdout.Stat()
	useColor = err == nil && st.Mode()&os.ModeCharDevice != 0
}

func c(code, s string) string {
	if !useColor {
		return s
	}
	return "\033[" + code + "m" + s + "\033[0m"
}

func green(s string) string  { return c("32", s) }
func yellow(s string) string { return c("33", s) }
func red(s string) string    { return c("31", s) }
func dim(s string) string    { return c("2", s) }
func bold(s string) string   { return c("1", s) }
func cyan(s string) string   { return c("36", s) }

const rule = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

func header(sub string) {
	fmt.Printf("\n%s %s\n%s\n", bold("Herd"), dim("— "+sub), dim(rule))
}

// --- sections -------------------------------------------------------------

func printHardware(hw Hardware) {
	fmt.Println(bold("Hardware detected:"))
	if len(hw.GPUs) == 0 {
		fmt.Printf("  %s  %s\n", dim("GPU: "), yellow("none — CPU inference only"))
	}
	for _, g := range hw.GPUs {
		bw := fmt.Sprintf("%.0f GB/s", g.BandwidthGBs)
		if !g.BandwidthKnown {
			bw = yellow(bw + "?")
		}
		fmt.Printf("  %s  %s (%.1fGB VRAM, %s available, %s, %s)\n", dim("GPU: "), bold(g.Name),
			g.VRAMTotalGB, avail(g.VRAMAvailableGB, g.VRAMTotalGB), bw, g.Backend)
	}
	fmt.Printf("  %s  %s — %d cores / %d threads\n", dim("CPU: "), hw.CPUName, hw.CPUCores, hw.CPUThreads)
	fmt.Printf("  %s  %.1fGB (%s available)\n", dim("RAM: "), hw.RAMTotalGB,
		avail(hw.RAMAvailableGB, hw.RAMTotalGB))
	fmt.Printf("  %s   %s\n", dim("OS: "), hw.OS)
}

// avail colours the free-memory figure by how much headroom is left.
func avail(free, total float64) string {
	s := fmt.Sprintf("%.1fGB", free)
	if total <= 0 {
		return s
	}
	switch r := free / total; {
	case r > 0.6:
		return green(s)
	case r > 0.25:
		return yellow(s)
	default:
		return red(s)
	}
}

func printFits(title, mark string, fits []Fit, paint func(string) string, showSpeed bool) {
	if len(fits) == 0 {
		return
	}
	fmt.Printf("\n%s\n", bold(paint(title)))
	for _, f := range fits {
		speed := dim("—")
		if showSpeed {
			speed = fmt.Sprintf("~%.0f tok/s", f.TokensPerSec)
		}
		extra := strings.Join(f.UseCases, ", ")
		if f.Tier == "offload" {
			extra = fmt.Sprintf("%d/%d layers on GPU · %s", f.GPULayers, f.TotalLayers, extra)
		}
		if f.Arch == "moe" {
			extra = "MoE · " + extra
		}
		fmt.Printf("  %s %-19s %-5s %6.1fGB  %-11s %s\n",
			paint(mark), f.Model, f.Quant, f.TotalGB, speed, dim(extra))
	}
}

func printHybrid(h Hybrid) {
	if len(h.Local) == 0 && len(h.Cloud) == 0 {
		return
	}
	fmt.Printf("\n%s\n", bold(cyan("HYBRID PLAN")))
	for _, a := range h.Local {
		fmt.Printf("  %s %-10s %s %s %s\n", green("local "), a.UseCase+":",
			bold(a.Model), dim(a.Quant),
			dim(fmt.Sprintf("~%.0f tok/s · %.0f%% of calls", a.TokensPerSec, a.Share*100)))
	}
	if len(h.Cloud) > 0 {
		fmt.Printf("  %s %s\n", yellow("cloud "), dim(strings.Join(h.Cloud, ", ")))
	}
	fmt.Printf("  %s %.0f%% of a typical mix stays local · saves ~$%.2f per 1000 calls (of $%.2f)\n",
		dim("→"), h.Coverage*100, h.Savings1k, h.CloudCost1k)
}

func printTips(tips []Tip, full bool) {
	if len(tips) == 0 {
		return
	}
	fmt.Printf("\n%s\n", bold(cyan("OPTIMIZATIONS")))
	for _, t := range tips {
		fmt.Printf("  %s %s", green("»"), bold(t.Title))
		if t.Impact != "" {
			fmt.Printf("  %s", green("["+t.Impact+"]"))
		}
		fmt.Println()
		if full {
			for _, line := range wrap(t.Detail, 78) {
				fmt.Printf("      %s\n", dim(line))
			}
			if t.Command != "" {
				fmt.Printf("      %s\n", cyan("$ "+t.Command))
			}
			fmt.Println()
		}
	}
}

func printNotes(notes []string) {
	for _, n := range notes {
		lines := wrap(n, 76)
		fmt.Printf("\n  %s %s\n", yellow("!"), lines[0])
		for _, l := range lines[1:] {
			fmt.Printf("    %s\n", dim(l))
		}
	}
}

func wrap(s string, width int) []string {
	words := strings.Fields(s)
	if len(words) == 0 {
		return []string{""}
	}
	var out []string
	line := words[0]
	for _, w := range words[1:] {
		if len(line)+1+len(w) > width {
			out = append(out, line)
			line = w
		} else {
			line += " " + w
		}
	}
	return append(out, line)
}

// OverheadOr reports the runtime overhead the backend charged, falling back to a
// default when an older backend omits the field.
func (f *Fit) OverheadOr(def float64) float64 {
	if f.OverheadGB > 0 {
		return f.OverheadGB
	}
	return def
}

// printBackends renders the survey: what works here, what each one costs, and what
// the scheduler can do with it.
func printBackends(bs []BackendInfo, active string) {
	fmt.Printf("\n%s\n", bold(cyan("BACKENDS")))
	for _, b := range bs {
		mark, tone := "✗", red
		if b.Available {
			mark, tone = "✓", green
		}
		name := b.Label
		if b.ID == active {
			name += " (in use)"
		}
		fmt.Printf("  %s %-28s %s\n", tone(mark), bold(name), dim(b.Detail))
		caps := []string{strings.Join(b.Quants, "/")}
		if b.CanEvict {
			caps = append(caps, "schedulable")
		} else {
			caps = append(caps, yellow("cannot evict"))
		}
		if b.OwnsCard {
			caps = append(caps, yellow("reserves the whole GPU"))
		}
		caps = append(caps, fmt.Sprintf("%.1fGB runtime", b.OverheadGB))
		fmt.Printf("      %s\n", dim(strings.Join(caps, " · ")))
		if !b.Available && !strings.Contains(b.Detail, b.Install) {
			fmt.Printf("      %s\n", dim(b.Install))
		} else if b.Available && len(b.Models) > 0 {
			shown := b.Models
			extra := ""
			if len(shown) > 4 {
				extra = fmt.Sprintf(" +%d more", len(shown)-4)
				shown = shown[:4]
			}
			fmt.Printf("      %s\n", dim("models: "+strings.Join(shown, ", ")+extra))
		}
	}
}
