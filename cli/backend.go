package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// The Go binary is a shell: it parses flags and draws output. Everything that needs
// to know about GPUs, VRAM or Ollama lives in the Python package and speaks JSON.

type backendError struct {
	Error string `json:"error"`
	Hint  string `json:"hint"`
}

// findPython locates an interpreter that can import the herd package. Order:
// $HERD_PYTHON, a .venv above the binary or the working directory, the path recorded
// by `make install`, then PATH. The recorded path is what makes an installed binary
// work from any directory without the user exporting anything.
func findPython() (string, error) {
	if p := os.Getenv("HERD_PYTHON"); p != "" {
		return p, nil
	}
	venvBin := "bin/python"
	names := []string{"python3", "python"}
	if runtime.GOOS == "windows" {
		venvBin = "Scripts/python.exe"
		names = []string{"python.exe", "python3.exe"}
	}

	var roots []string
	if exe, err := os.Executable(); err == nil {
		roots = append(roots, filepath.Dir(exe))
	}
	if wd, err := os.Getwd(); err == nil {
		roots = append(roots, wd)
	}
	seen := map[string]bool{}
	for _, root := range roots {
		for dir := root; dir != filepath.Dir(dir); dir = filepath.Dir(dir) {
			if seen[dir] {
				break
			}
			seen[dir] = true
			cand := filepath.Join(dir, ".venv", venvBin)
			if st, err := os.Stat(cand); err == nil && !st.IsDir() {
				return cand, nil
			}
		}
	}
	if p := recordedPython(); p != "" {
		return p, nil
	}
	for _, n := range names {
		if p, err := exec.LookPath(n); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("no Python interpreter found")
}

// recordedPython reads the interpreter `make install` wrote down, if it still exists.
func recordedPython() string {
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		dir = filepath.Join(home, ".config")
	}
	b, err := os.ReadFile(filepath.Join(dir, "herd", "python"))
	if err != nil {
		return ""
	}
	p := strings.TrimSpace(string(b))
	if st, err := os.Stat(p); err != nil || st.IsDir() {
		return ""
	}
	return p
}

func backendCmd(args ...string) (*exec.Cmd, error) {
	py, err := findPython()
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(py, append([]string{"-m", "herd"}, args...)...)
	// Let the backend be found when the binary is run from outside the source tree.
	if root := os.Getenv("HERD_ROOT"); root != "" {
		cmd.Env = append(os.Environ(), "PYTHONPATH="+root+string(os.PathListSeparator)+os.Getenv("PYTHONPATH"))
	}
	return cmd, nil
}

// call runs a backend command that returns exactly one JSON object.
func call(out any, args ...string) error {
	cmd, err := backendCmd(args...)
	if err != nil {
		return installHint(err)
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	stdout, err := cmd.Output()
	if len(stdout) > 0 {
		var be backendError
		if json.Unmarshal(stdout, &be) == nil && be.Error != "" {
			msg := be.Error
			if be.Hint != "" {
				msg += "\n  " + be.Hint
			}
			return fmt.Errorf("%s", msg)
		}
	}
	if err != nil {
		e := strings.TrimSpace(stderr.String())
		if strings.Contains(e, "No module named herd") {
			return installHint(fmt.Errorf("the herd Python backend is not installed"))
		}
		if e == "" {
			e = err.Error()
		}
		return fmt.Errorf("backend failed: %s", lastLines(e, 4))
	}
	return json.Unmarshal(stdout, out)
}

// stream runs a backend command that emits NDJSON, handing each object to fn.
func stream(fn func(map[string]any), args ...string) error {
	cmd, err := backendCmd(args...)
	if err != nil {
		return installHint(err)
	}
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	sc := bufio.NewScanner(pipe)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	var backendErr error
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var obj map[string]any
		if json.Unmarshal([]byte(line), &obj) != nil {
			continue
		}
		// A stream can end in a structured error instead of a final event.
		if msg, ok := obj["error"].(string); ok && obj["event"] == nil {
			if hint, ok := obj["hint"].(string); ok && hint != "" {
				msg += "\n  " + hint
			}
			backendErr = fmt.Errorf("%s", msg)
			continue
		}
		fn(obj)
	}
	if err := cmd.Wait(); err != nil {
		if backendErr != nil {
			return backendErr
		}
		if e := strings.TrimSpace(stderr.String()); e != "" {
			return fmt.Errorf("backend failed: %s", lastLines(e, 4))
		}
		return err
	}
	return nil
}

// passthrough runs a backend command with stdio attached — used for `serve`.
func passthrough(args ...string) error {
	cmd, err := backendCmd(args...)
	if err != nil {
		return installHint(err)
	}
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

// raw prints the backend's JSON verbatim for --json.
func raw(w io.Writer, args ...string) error {
	var v json.RawMessage
	if err := call(&v, args...); err != nil {
		return err
	}
	var buf bytes.Buffer
	if err := json.Indent(&buf, v, "", "  "); err != nil {
		return err
	}
	fmt.Fprintln(w, buf.String())
	return nil
}

func installHint(err error) error {
	return fmt.Errorf("%v\n\n  Herd needs its Python backend. From the project root:\n"+
		"    uv venv && uv pip install -e .\n"+
		"  Or point at an interpreter that has it: export HERD_PYTHON=/path/to/python", err)
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n  ")
}
