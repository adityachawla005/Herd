package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"time"
)

const version = "0.1.0"

const usage = `Herd — local LLM hardware profiler and multi-agent orchestrator

Usage:
  herd <command> [flags]

Commands:
  detect      Profile this machine: GPU, VRAM, CPU, RAM, memory bandwidth
  recommend   Which local models run well here, and how to make them run better
  optimize    Optimization playbook, tuned to one model if you name it
  models      List the model registry
  backends    Which inference backends work here (Ollama, llama.cpp, HF, vLLM)
  run         Run a task on the best local model for it
  bench       Measure this machine's real decode speed and calibrate the estimates
  agent       Route a task through the multi-agent orchestrator
  status      What is resident in VRAM right now, and the local-vs-cloud tally
  dashboard   Open the web UI with the live VRAM monitor
  serve       Start the web UI
  version     Print the version

Run 'herd <command> -h' for the flags of a command.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "detect":
		err = cmdDetect(os.Args[2:])
	case "recommend":
		err = cmdRecommend(os.Args[2:])
	case "optimize":
		err = cmdOptimize(os.Args[2:])
	case "models":
		err = cmdModels(os.Args[2:])
	case "backends":
		err = cmdBackends(os.Args[2:])
	case "run":
		err = cmdRun(os.Args[2:])
	case "agent":
		err = cmdAgent(os.Args[2:])
	case "status":
		err = cmdStatus(os.Args[2:])
	case "dashboard":
		err = cmdDashboard(os.Args[2:])
	case "bench":
		err = cmdBench(os.Args[2:])
	case "serve":
		err = cmdServe(os.Args[2:])
	case "version", "--version", "-v":
		fmt.Printf("herd %s\n", version)
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
	if err != nil {
		initColor(false)
		fmt.Fprintf(os.Stderr, "\n%s %v\n\n", red("error:"), err)
		os.Exit(1)
	}
}

// flagset wires the flags every profiling command shares.
type common struct {
	jsonOut bool
	noColor bool
	context int
	kvQuant string
	backend string
}

func newFlags(name string, c *common, withContext bool) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ExitOnError)
	fs.BoolVar(&c.jsonOut, "json", false, "emit raw JSON")
	fs.BoolVar(&c.noColor, "no-color", false, "disable colour output")
	if withContext {
		fs.IntVar(&c.context, "context", 0, "context length to size the KV cache for (default 4096)")
		fs.StringVar(&c.kvQuant, "kv-quant", "fp16", "KV cache precision: fp16, q8, q4")
	}
	fs.StringVar(&c.backend, "backend", "", "ollama, llamacpp, hf or openai (default: first available)")
	return fs
}

func (c *common) args(base ...string) []string {
	if c.context > 0 {
		base = append(base, "--context", fmt.Sprint(c.context))
	}
	if c.kvQuant != "" && c.kvQuant != "fp16" {
		base = append(base, "--kv-quant", c.kvQuant)
	}
	if c.backend != "" {
		base = append(base, "--backend", c.backend)
	}
	return base
}

func cmdDetect(argv []string) error {
	var c common
	fs := newFlags("detect", &c, false)
	fs.Parse(argv)
	initColor(c.noColor)
	if c.jsonOut {
		return raw(os.Stdout, "detect")
	}
	var hw Hardware
	if err := call(&hw, "detect"); err != nil {
		return err
	}
	header("hardware profile")
	printHardware(hw)
	printNotes(hw.Notes)
	fmt.Printf("\n%s\n\n", dim("Next: herd recommend"))
	return nil
}

func cmdRecommend(argv []string) error {
	var c common
	var limit int
	var verbose bool
	fs := newFlags("recommend", &c, true)
	fs.IntVar(&limit, "limit", 8, "max rows per section")
	fs.BoolVar(&verbose, "verbose", false, "show the full text of each optimization")
	fs.Parse(argv)
	initColor(c.noColor)

	args := c.args("recommend", "--limit", fmt.Sprint(limit))
	if c.jsonOut {
		return raw(os.Stdout, args...)
	}
	var r Recommendation
	if err := call(&r, args...); err != nil {
		return err
	}
	header(fmt.Sprintf("%s · %d ctx · %s KV cache", backendLabel(r.Backends, r.Backend),
		r.Context, r.KVQuant))
	printHardware(r.Hardware)
	printFits("RUNS GREAT (fully in VRAM)", "✓", r.RunsGreat, green, true)
	printFits("RUNS (CPU offload, slower)", "~", r.RunsOffload, yellow, true)
	printFits("WON'T FIT (even at Q4)", "✗", r.WontFit, red, false)
	printHybrid(r.Hybrid)
	printTips(r.Optimizations, verbose)
	if verbose {
		printBackends(r.Backends, r.Backend)
	} else if n := countAvailable(r.Backends); n > 1 {
		fmt.Printf("\n%s\n", dim(fmt.Sprintf("%d backends available — `herd backends` to compare, "+
			"`--backend <id>` to size for another", n)))
	}
	printNotes(r.Notes)
	cal := dim(fmt.Sprintf("uncalibrated — speeds assume %.0fms/token overhead; run `herd bench` to measure",
		r.Calibration.TokenOverheadMS))
	switch {
	case r.Calibration.Measured:
		detail := fmt.Sprintf("%.0fms/token", r.Calibration.TokenOverheadMS)
		if r.Calibration.EffectiveBW != nil {
			detail += fmt.Sprintf(" at a measured %.0f GB/s", *r.Calibration.EffectiveBW)
		}
		cal = dim("calibrated: " + detail + " on this GPU")
	case r.Calibration.Unreliable:
		cal = yellow("speeds below are unvalidated: " + r.Calibration.Reason)
	}
	fmt.Printf("\n%s\n", cal)
	if !verbose {
		fmt.Printf("%s\n", dim("Full detail and commands: herd optimize --model <name>"))
	}
	fmt.Println()
	return nil
}

func cmdOptimize(argv []string) error {
	var c common
	var model, quant string
	fs := newFlags("optimize", &c, true)
	fs.StringVar(&model, "model", "", "tune the advice to one model (e.g. llama3:8b)")
	fs.StringVar(&quant, "quant", "Q4", "quantization: Q4, Q5, Q8, FP16")
	fs.Parse(argv)
	initColor(c.noColor)

	args := c.args("optimize", "--quant", quant)
	if model != "" {
		args = append(args, "--model", model)
	}
	if c.jsonOut {
		return raw(os.Stdout, args...)
	}
	var r OptimizeResult
	if err := call(&r, args...); err != nil {
		return err
	}
	sub := "optimization playbook"
	if r.Fit != nil {
		sub = fmt.Sprintf("%s %s @ %d ctx", r.Fit.Model, r.Fit.Quant, r.Fit.Context)
	}
	header(sub)
	if f := r.Fit; f != nil {
		verdict := green("fits entirely in VRAM")
		if !f.FitsVRAM && f.FitsOffload {
			verdict = yellow(fmt.Sprintf("needs CPU offload — %d of %d layers on GPU", f.GPULayers, f.TotalLayers))
		} else if !f.FitsOffload {
			verdict = red("does not fit, even with RAM offload")
		}
		fmt.Printf("  %-9s %.1fGB weights + %.2fGB KV + %.1fGB runtime = %s\n",
			dim("memory:"), f.WeightGB, f.KVGB, f.OverheadOr(0.5), bold(fmt.Sprintf("%.1fGB", f.TotalGB)))
		fmt.Printf("  %-9s %s\n", dim("verdict:"), verdict)
		fmt.Printf("  %-9s ~%.0f tok/s\n", dim("speed:"), f.TokensPerSec)
		for _, n := range f.Notes {
			for i, l := range wrap(n, 76) {
				if i == 0 {
					fmt.Printf("  %-9s %s\n", dim("note:"), l)
				} else {
					fmt.Printf("            %s\n", dim(l))
				}
			}
		}
	} else {
		printHardware(r.Hardware)
		fmt.Printf("\n%s\n", dim("Tip: pass --model <name> for advice sized to a specific model."))
	}
	printTips(r.Optimizations, true)
	printNotes(r.Notes)
	return nil
}

func cmdBackends(argv []string) error {
	var c common
	fs := newFlags("backends", &c, false)
	fs.Parse(argv)
	initColor(c.noColor)
	if c.jsonOut {
		return raw(os.Stdout, "backends")
	}
	var r struct {
		Backends []BackendInfo `json:"backends"`
		Default  string        `json:"default"`
	}
	if err := call(&r, "backends"); err != nil {
		return err
	}
	header("inference backends")
	printBackends(r.Backends, r.Default)
	if r.Default == "" {
		fmt.Printf("\n  %s %s\n", red("✗"), "Nothing usable here — install Ollama to start.")
	}
	fmt.Println()
	return nil
}

func backendLabel(bs []BackendInfo, id string) string {
	for _, b := range bs {
		if b.ID == id {
			return b.Label
		}
	}
	return id
}

func countAvailable(bs []BackendInfo) int {
	n := 0
	for _, b := range bs {
		if b.Available {
			n++
		}
	}
	return n
}

func cmdModels(argv []string) error {
	var c common
	fs := newFlags("models", &c, false)
	fs.Parse(argv)
	initColor(c.noColor)
	if c.jsonOut {
		return raw(os.Stdout, "models")
	}
	var reg struct {
		Models []struct {
			Name     string   `json:"name"`
			Ollama   string   `json:"ollama"`
			ParamsB  float64  `json:"params_b"`
			Arch     string   `json:"arch"`
			Layers   int      `json:"layers"`
			MaxCtx   int      `json:"max_context"`
			Quants   []string `json:"quants"`
			UseCases []string `json:"use_cases"`
		} `json:"models"`
	}
	if err := call(&reg, "models"); err != nil {
		return err
	}
	header(fmt.Sprintf("model registry (%d models)", len(reg.Models)))
	fmt.Printf("  %-20s %-20s %6s %-5s %8s  %s\n",
		dim("NAME"), dim("OLLAMA TAG"), dim("PARAMS"), dim("ARCH"), dim("MAX CTX"), dim("USE CASES"))
	for _, m := range reg.Models {
		arch := m.Arch
		if arch == "moe" {
			arch = cyan(arch)
		}
		fmt.Printf("  %-20s %-20s %5.1fB %-5s %8d  %s\n", m.Name, dim(m.Ollama), m.ParamsB,
			arch, m.MaxCtx, dim(strings.Join(m.UseCases, ", ")))
	}
	fmt.Printf("\n%s\n\n", dim("Edit herd/models.json to add models — no rebuild needed."))
	return nil
}

// cmdRun streams a generation. Tokens go to stdout so the output pipes cleanly;
// the selection banner and stats go to stderr.
func cmdRun(argv []string) error {
	var c common
	var taskType, model, file, keepAlive string
	var quiet bool
	fs := newFlags("run", &c, true)
	fs.StringVar(&taskType, "task-type", "", "chat, code, fast or reasoning (default: inferred)")
	fs.StringVar(&model, "model", "", "force a specific Ollama model, e.g. llama3:8b")
	fs.StringVar(&file, "file", "", "append a file to the prompt")
	fs.StringVar(&keepAlive, "keep-alive", "", "how long to keep the model in VRAM, e.g. 30m or 0")
	fs.BoolVar(&quiet, "quiet", false, "tokens only — no banner or stats")
	fs.Parse(argv)
	initColor(c.noColor)

	task := strings.TrimSpace(strings.Join(fs.Args(), " "))
	if task == "" {
		return fmt.Errorf("nothing to run\n\n  herd run \"explain how async/await works in Python\"")
	}

	args := c.args("run", task)
	for _, kv := range [][2]string{{"--task-type", taskType}, {"--model", model},
		{"--file", file}, {"--keep-alive", keepAlive}} {
		if kv[1] != "" {
			args = append(args, kv[0], kv[1])
		}
	}

	started := false
	err := stream(func(e map[string]any) {
		switch e["event"] {
		case "selection":
			if quiet {
				return
			}
			fmt.Fprintf(os.Stderr, "\n%s %s  %s\n", green("▸"), bold(str(e["model"])),
				dim(str(e["tag"])))
			fmt.Fprintf(os.Stderr, "  %s %s · %s\n\n", dim("why:"), cyan(str(e["task_type"])),
				dim(str(e["reason"])))
		case "token":
			started = true
			fmt.Print(str(e["text"]))
		case "done":
			if started {
				fmt.Println()
			}
			if quiet {
				return
			}
			if s, ok := e["stats"].(map[string]any); ok {
				fmt.Fprintf(os.Stderr, "\n%s\n", dim(fmt.Sprintf(
					"%s tokens · %s tok/s · %s to first token · %s load",
					num(s["tokens"], 0), num(s["tokens_per_sec"], 1),
					ms(s["ttft_ms"]), ms(s["load_ms"]))))
			}
		}
	}, args...)
	if err != nil {
		return err
	}
	return nil
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func num(v any, places int) string {
	f, ok := v.(float64)
	if !ok {
		return "?"
	}
	return fmt.Sprintf("%.*f", places, f)
}

func ms(v any) string {
	f, ok := v.(float64)
	if !ok {
		return "?"
	}
	if f >= 1000 {
		return fmt.Sprintf("%.1fs", f/1000)
	}
	return fmt.Sprintf("%.0fms", f)
}

// cmdAgent renders the routing decisions and scheduler activity around a generation.
func cmdAgent(argv []string) error {
	var c common
	var agent string
	var verbose, local, quiet bool
	fs := newFlags("agent", &c, false)
	fs.StringVar(&agent, "agent", "", "skip the router and use this agent")
	fs.BoolVar(&verbose, "verbose", false, "show scheduler load/evict activity live")
	fs.BoolVar(&local, "local", false, "never route to the cloud")
	fs.BoolVar(&quiet, "quiet", false, "tokens only")
	fs.Parse(argv)
	initColor(c.noColor)

	task := strings.TrimSpace(strings.Join(fs.Args(), " "))
	if task == "" {
		return fmt.Errorf("nothing to run\n\n  herd agent \"write a REST API in Go\"")
	}
	args := []string{"agent", task}
	if agent != "" {
		args = append(args, "--agent", agent)
	}
	if local {
		args = append(args, "--local")
	}

	log := func(format string, a ...any) {
		if !quiet {
			fmt.Fprintf(os.Stderr, format, a...)
		}
	}
	// Scheduler activity for the router arrives before the routing decision itself,
	// so label the section up front rather than leaving a bare cache line at the top.
	if verbose {
		log("\n%s\n", dim("◆ routing…"))
	}
	started := false
	return stream(func(e map[string]any) {
		switch e["event"] {
		case "route":
			cx := num(e["complexity"], 0)
			prefix := "\n"
			if verbose {
				prefix = ""
			}
			log("%s%s %s %s\n", prefix, cyan("◆ router"), bold(str(e["agent"])),
				dim(fmt.Sprintf("· complexity %s/10 · %s", cx, str(e["reason"]))))
		case "dispatch":
			dest, tone := "local", green
			if str(e["destination"]) == "cloud" {
				dest, tone = "cloud", yellow
				log("  %s %s\n", yellow("⚠ routing to cloud —"), dim(str(e["reason"])))
			}
			via := ""
			if b := str(e["backend"]); b != "" && dest == "local" {
				via = "via " + b + " · "
			}
			log("%s %s %s\n\n", tone("▸ "+dest), bold(str(e["model"])), dim(via+str(e["reason"])))
			_ = dest
		case "load":
			if verbose {
				log("  %s %s %s\n", dim("↑ load "), str(e["tag"]),
					dim(fmt.Sprintf("%sGB · %sms", num(e["size_gb"], 2), num(e["load_ms"], 0))))
			}
		case "cache_hit":
			if verbose {
				log("  %s %s %s\n", green("✓ cached"), str(e["tag"]),
					dim(fmt.Sprintf("%sGB already resident", num(e["size_gb"], 2))))
			}
		case "evict":
			if verbose {
				log("  %s %s %s\n", yellow("↓ evict"), str(e["tag"]),
					dim(fmt.Sprintf("freed %sGB after %ss idle",
						num(e["freed_gb"], 2), num(e["idle_s"], 0))))
			}
		case "pressure":
			log("  %s %s\n", yellow("! vram pressure"),
				dim(fmt.Sprintf("%s needs %sGB, only %sGB free — nothing evictable",
					str(e["tag"]), num(e["need_gb"], 2), num(e["free_gb"], 2))))
		case "token":
			started = true
			fmt.Print(str(e["text"]))
		case "done":
			if started {
				fmt.Println()
			}
			s, _ := e["stats"].(map[string]any)
			tot, _ := e["totals"].(map[string]any)
			if s == nil {
				return
			}
			line := fmt.Sprintf("%s tokens", num(s["tokens"], 0))
			if s["tokens_per_sec"] != nil {
				line += fmt.Sprintf(" · %s tok/s", num(s["tokens_per_sec"], 1))
			}
			if hit, ok := s["cache_hit"].(bool); ok {
				line += map[bool]string{true: " · cache hit", false: " · cold load"}[hit]
			}
			if s["cost_usd"] != nil {
				line += fmt.Sprintf(" · $%s", num(s["cost_usd"], 5))
			}
			if tot != nil {
				line += fmt.Sprintf(" · %s local / %s cloud so far, ~$%s saved",
					num(tot["local_tasks"], 0), num(tot["cloud_tasks"], 0),
					num(tot["estimated_saved_usd"], 2))
			}
			log("\n%s\n", dim(line))
		}
	}, args...)
}

func cmdStatus(argv []string) error {
	var c common
	fs := newFlags("status", &c, false)
	fs.Parse(argv)
	initColor(c.noColor)
	if c.jsonOut {
		return raw(os.Stdout, "status")
	}
	var st struct {
		VRAM struct {
			TotalGB     float64 `json:"vram_total_gb"`
			BudgetGB    float64 `json:"budget_gb"`
			UsedGB      float64 `json:"used_gb"`
			FreeGB      float64 `json:"free_gb"`
			Utilization float64 `json:"utilization"`
			Loaded      []struct {
				Tag     string   `json:"tag"`
				SizeGB  float64  `json:"size_gb"`
				IdleS   float64  `json:"idle_s"`
				Pinned  bool     `json:"pinned"`
				Holders []string `json:"holders"`
			} `json:"loaded"`
		} `json:"vram"`
		Agents map[string]struct {
			Tag      string   `json:"tag"`
			Why      string   `json:"why"`
			Pinned   bool     `json:"pinned"`
			Resident bool     `json:"resident"`
			TokS     *float64 `json:"est_tok_s"`
		} `json:"agents"`
		Totals struct {
			LocalTasks int     `json:"local_tasks"`
			CloudTasks int     `json:"cloud_tasks"`
			Saved      float64 `json:"estimated_saved_usd"`
			Spend      float64 `json:"cloud_spend_usd"`
		} `json:"totals"`
		Calibrated bool `json:"calibrated"`
	}
	if err := call(&st, "status"); err != nil {
		return err
	}
	header("scheduler status")

	v := st.VRAM
	width := 40
	filled := int(v.Utilization * float64(width))
	if filled > width {
		filled = width
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("·", width-filled)
	tone := green
	if v.Utilization > 0.85 {
		tone = red
	} else if v.Utilization > 0.6 {
		tone = yellow
	}
	fmt.Printf("  %s %s\n", tone(bar),
		fmt.Sprintf("%.1f / %.1fGB used", v.UsedGB, v.BudgetGB))
	fmt.Printf("  %s\n\n", dim(fmt.Sprintf("%.1fGB card · %.1fGB free to the scheduler",
		v.TotalGB, v.FreeGB)))

	if len(v.Loaded) == 0 {
		fmt.Printf("  %s\n", dim("nothing resident"))
	}
	for _, m := range v.Loaded {
		flags := []string{}
		if m.Pinned {
			flags = append(flags, "pinned")
		}
		if len(m.Holders) > 0 {
			flags = append(flags, "in use by "+strings.Join(m.Holders, ", "))
		} else {
			flags = append(flags, fmt.Sprintf("idle %.0fs", m.IdleS))
		}
		fmt.Printf("  %s %-22s %6.2fGB  %s\n", green("●"), m.Tag, m.SizeGB,
			dim(strings.Join(flags, " · ")))
	}

	fmt.Printf("\n%s\n", bold(cyan("AGENTS")))
	for _, name := range sortedKeys(st.Agents) {
		a := st.Agents[name]
		dot, tag := dim("○"), a.Tag
		if a.Resident {
			dot = green("●")
		}
		if tag == "" {
			tag = red("unavailable")
		} else if strings.HasPrefix(tag, "cloud/") {
			tag = yellow(tag)
		}
		speed := ""
		if a.TokS != nil {
			speed = fmt.Sprintf("~%.0f tok/s", *a.TokS)
		}
		fmt.Printf("  %s %-12s %-26s %-11s %s\n", dot, name, tag, dim(speed), dim(a.Why))
	}

	t := st.Totals
	fmt.Printf("\n%s %d local · %d cloud · %s\n", bold(cyan("TASKS")), t.LocalTasks, t.CloudTasks,
		green(fmt.Sprintf("~$%.2f saved", t.Saved)))
	if t.Spend > 0 {
		fmt.Printf("  %s\n", dim(fmt.Sprintf("cloud spend so far: $%.4f", t.Spend)))
	}
	if !st.Calibrated {
		fmt.Printf("\n  %s\n", dim("speeds are uncalibrated — run `herd bench`"))
	}
	fmt.Println()
	return nil
}

func cmdDashboard(argv []string) error {
	fs := flag.NewFlagSet("dashboard", flag.ExitOnError)
	port := fs.Int("port", 8787, "port")
	fs.Parse(argv)
	initColor(false)
	url := fmt.Sprintf("http://127.0.0.1:%d/", *port)
	go func() {
		// Give uvicorn a moment to bind before the browser asks for the page.
		time.Sleep(1200 * time.Millisecond)
		open(url)
	}()
	fmt.Printf("\n%s %s\n%s\n\n", bold("Herd"), dim("— dashboard on "+url), dim("ctrl-c to stop"))
	return passthrough("serve", "--host", "127.0.0.1", "--port", fmt.Sprint(*port))
}

func open(url string) {
	cmds := map[string][]string{
		"darwin":  {"open", url},
		"windows": {"rundll32", "url.dll,FileProtocolHandler", url},
	}
	c, ok := cmds[runtime.GOOS]
	if !ok {
		c = []string{"xdg-open", url}
	}
	_ = exec.Command(c[0], c[1:]...).Start()
}

func sortedKeys[T any](m map[string]T) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func cmdBench(argv []string) error {
	var c common
	var model string
	var tokens int
	fs := newFlags("bench", &c, false)
	fs.StringVar(&model, "model", "", "model to benchmark (default: every one that fits)")
	fs.IntVar(&tokens, "tokens", 128, "tokens to generate per model")
	fs.Parse(argv)
	initColor(c.noColor)

	args := []string{"bench", "--tokens", fmt.Sprint(tokens)}
	if model != "" {
		args = append(args, "--model", model)
	}
	if c.jsonOut {
		return raw(os.Stdout, args...)
	}
	var b struct {
		Device   string  `json:"device"`
		TableBW  float64 `json:"table_bandwidth_gbs"`
		Degraded bool    `json:"driver_degraded"`
		Samples  []struct {
			Model      string  `json:"model"`
			TokS       float64 `json:"tok_s"`
			ResidentGB float64 `json:"resident_gb"`
			FullyOnGPU bool    `json:"fully_on_gpu"`
		} `json:"samples"`
		Fit struct {
			OK          bool    `json:"ok"`
			BandwidthBW float64 `json:"effective_bandwidth_gbs"`
			OverheadMS  float64 `json:"token_overhead_ms"`
			Reason      string  `json:"reason"`
		} `json:"fit"`
		OverheadMS float64 `json:"token_overhead_ms"`
		SavedTo    string  `json:"saved_to"`
		RejectedTo string  `json:"rejected_to"`
		Warning    string  `json:"warning"`
	}
	fmt.Fprintf(os.Stderr, "%s\n", dim("timing each model that fits, on an otherwise empty card…"))
	if err := call(&b, args...); err != nil {
		return err
	}
	header("speed calibration")
	fmt.Printf("  %s %s\n\n", dim("device:"), b.Device)
	fmt.Printf("  %-22s %10s %12s %10s\n", dim("MODEL"), dim("RESIDENT"), dim("MEASURED"),
		dim("BANDWIDTH-ONLY"))
	for _, s := range b.Samples {
		pure := b.TableBW / s.ResidentGB
		flag := ""
		if !s.FullyOnGPU {
			flag = yellow(" (partly on CPU)")
		}
		fmt.Printf("  %-22s %8.2fGB %8s tok/s %9s%s\n", s.Model, s.ResidentGB,
			green(fmt.Sprintf("%.1f", s.TokS)), dim(fmt.Sprintf("%.0f", pure)), flag)
	}

	fmt.Println()
	if b.Fit.OK || b.OverheadMS > 0 {
		if b.Fit.BandwidthBW > 0 {
			fmt.Printf("  %-18s %s %s\n", dim("measured bandwidth:"),
				bold(fmt.Sprintf("%.0f GB/s", b.Fit.BandwidthBW)),
				dim(fmt.Sprintf("(rated %.0f)", b.TableBW)))
		}
		fmt.Printf("  %-18s %s\n", dim("fixed cost:"),
			bold(fmt.Sprintf("%.1f ms/token", b.OverheadMS)))
	}
	if b.Warning != "" {
		printNotes([]string{b.Warning})
	}
	if b.Degraded {
		printNotes([]string{"The NVIDIA driver could not report clocks or power during this " +
			"run, so these timings describe a card in an unknown state."})
	}
	if b.SavedTo != "" {
		fmt.Printf("\n  %s %s\n", green("✓"), dim("saved to "+b.SavedTo))
	} else if b.RejectedTo != "" {
		fmt.Printf("\n  %s %s\n", yellow("!"),
			dim("recorded as unreliable in "+b.RejectedTo+" — estimates stay on rated bandwidth"))
	}
	fmt.Println()
	return nil
}

func cmdServe(argv []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	host := fs.String("host", "127.0.0.1", "bind address")
	port := fs.Int("port", 8787, "port")
	fs.Parse(argv)
	initColor(false)
	fmt.Printf("\n%s %s\n", bold("Herd"), dim("— web UI on http://"+*host+":"+fmt.Sprint(*port)))
	fmt.Printf("%s\n\n", dim("ctrl-c to stop"))
	return passthrough("serve", "--host", *host, "--port", fmt.Sprint(*port))
}
