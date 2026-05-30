#!/usr/bin/env python3
"""Standalone animated skin preview — requires only `rich`. No Hermes setup needed.

Usage:
    python scripts/preview_skin.py                    # preview default skin
    python scripts/preview_skin.py ares               # preview one skin
    python scripts/preview_skin.py default hermes ares # preview multiple
    python scripts/preview_skin.py --all              # preview every built-in skin
    python scripts/preview_skin.py --no-animate       # skip animation, static only
"""

import itertools
import shutil
import sys
import time
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Stub hermes_constants so skin_engine imports without a Hermes installation.
# ---------------------------------------------------------------------------
_stub = types.ModuleType("hermes_constants")
_stub.get_hermes_home = lambda: Path.home() / ".hermes"
_stub.display_hermes_home = lambda: "~/.hermes"
sys.modules.setdefault("hermes_constants", _stub)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_cli.skin_engine import _BUILTIN_SKINS, _build_skin_config  # noqa: E402

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
except ImportError:
    sys.exit("rich is required:  pip install rich")

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Default art (fallback when skin has no custom logo/hero)
# ---------------------------------------------------------------------------

_DEFAULT_LOGO = """\
[bold #FFD700]██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #FFD700]██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#FFBF00]███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#FFBF00]██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#CD7F32]██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#CD7F32]╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]"""

_DEFAULT_HERO = """\
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀[/]
[#FFBF00]⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]"""

# Braille dot frames — same as KawaiiSpinner SPINNERS['dots']
_DOTS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

# Default thinking faces/verbs used when a skin has none of its own
_DEFAULT_THINKING_FACES = [
    "(｡•́︿•̀｡)", "(◔_◔)", "(¬‿¬)", "( •_•)>⌐■-■", "(⌐■_■)",
    "(´･_･`)", "◉_◉", "(°ロ°)",
]
_DEFAULT_THINKING_VERBS = [
    "pondering", "contemplating", "musing", "cogitating",
    "ruminating", "deliberating", "mulling", "processing",
]

# Mock data
_MOCK_VERSION = "v0.15.1 (2026-03-14)"
_MOCK_MODEL   = "claude-sonnet-4-6"
_MOCK_CTX     = "200K context"
_MOCK_CWD     = "~/projects/myapp"
_MOCK_SESSION = "a3f9b2c1"
_MOCK_TOOLS = {
    "terminal": ["terminal", "read_file", "write_file", "patch", "search_files"],
    "web":      ["web_search", "web_extract", "browser_navigate"],
    "memory":   ["memory_store", "memory_search"],
    "skills":   ["skill_run", "skill_manage"],
    "delegate": ["delegate_task"],
}
_MOCK_SKILLS = {
    "github":  ["pr-review", "commit-audit"],
    "mlops":   ["train-monitor", "eval-harness"],
    "devops":  ["deploy-check", "docker-compose"],
}
_MOCK_PROMPT   = "show me the project structure and summarise what this codebase does"
_MOCK_RESPONSE = """\
Here's what I found in your project:

  src/          — core application logic (12 files)
  tests/        — pytest suite, ~400 tests
  scripts/      — build and utility scripts
  docs/         — Docusaurus site

This is a Python CLI application with a plugin architecture. \
The main entry point is src/main.py, which bootstraps the \
plugin loader and dispatches to subcommands. \
Dependencies are pinned with upper bounds in pyproject.toml.\
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def swatch(hex_color: str) -> Text:
    t = Text()
    t.append("  ██  ", style=f"on {hex_color}")
    return t


def color_row(c: dict, key_a: str, label_a: str, key_b: str = None, label_b: str = None) -> None:
    val_a = c.get(key_a, "#888888")
    t = Text("  ")
    t.append_text(swatch(val_a))
    t.append(f" {label_a:<22}", style="dim")
    t.append(f" {val_a}", style=val_a)
    if key_b:
        val_b = c.get(key_b, "#888888")
        t.append("      ")
        t.append_text(swatch(val_b))
        t.append(f" {label_b:<22}", style="dim")
        t.append(f" {val_b}", style=val_b)
    console.print(t)


def section(label: str, accent: str) -> None:
    console.print(f"\n  [bold {accent}]{label}[/]\n")


# ---------------------------------------------------------------------------
# Animation: logo reveal line-by-line
# ---------------------------------------------------------------------------

def animate_logo(logo: str) -> None:
    for line in logo.splitlines():
        console.print(line)
        time.sleep(0.04)


# ---------------------------------------------------------------------------
# Animation: spinner + activity feed (mirrors KawaiiSpinner._animate)
# ---------------------------------------------------------------------------

def animate_spinner(skin, duration: float = 4.0) -> None:
    c = skin.colors
    prefix = skin.tool_prefix

    accent  = c.get("banner_accent", "#aaaaaa")
    dim     = c.get("banner_dim",    "#555555")
    title_c = c.get("banner_title",  "#ffffff")
    text_c  = c.get("banner_text",   "#dddddd")

    sp      = skin.spinner
    faces   = sp.get("thinking_faces", _DEFAULT_THINKING_FACES)
    verbs   = sp.get("thinking_verbs", _DEFAULT_THINKING_VERBS)
    wings   = skin.get_spinner_wings()

    face_cycle = itertools.cycle(faces)
    verb_cycle = itertools.cycle(verbs)
    dot_cycle  = itertools.cycle(_DOTS)
    wing_cycle = itertools.cycle(wings) if wings else None

    mock_activity = [
        ("terminal",    "ls -la ~/projects/myapp"),
        ("read_file",   "src/main.py:1-80"),
        ("search_files","pattern='class.*Plugin'"),
        ("web_search",  "plugin architecture python best practices"),
        ("read_file",   "pyproject.toml"),
    ]

    feed: list[tuple[str, str]] = []
    start          = time.time()
    frame_idx      = 0
    next_activity  = 0.9
    current_verb   = verbs[0]
    current_face   = faces[0]
    current_wing   = wings[0] if wings else None

    def build_frame(elapsed: float) -> Text:
        dot = next(dot_cycle)
        t   = Text()
        # Real spinner: plain unstyled text, no ANSI color (KawaiiSpinner._animate)
        if current_wing:
            lw, rw = current_wing
            t.append(f"  {lw} {dot} {current_face} {current_verb} {rw} ({elapsed:.1f}s)\n")
        else:
            t.append(f"  {dot} {current_face} {current_verb} ({elapsed:.1f}s)\n")

        # Tool activity lines: also plain in real CLI (printed via print_above)
        for tool, arg in feed[-5:]:
            t.append(f"  {prefix} {tool} → {arg}\n")

        return t

    with Live(build_frame(0), console=console, refresh_per_second=10) as live:
        while True:
            elapsed = time.time() - start
            if elapsed >= duration:
                break

            dot = _DOTS[frame_idx % len(_DOTS)]

            if frame_idx % 4 == 0:
                current_face = next(face_cycle)
                current_verb = next(verb_cycle)
                if wing_cycle:
                    current_wing = next(wing_cycle)

            if elapsed >= next_activity and len(feed) < len(mock_activity):
                feed.append(mock_activity[len(feed)])
                next_activity = elapsed + 0.75

            live.update(build_frame(elapsed))
            frame_idx += 1
            time.sleep(0.12)


# ---------------------------------------------------------------------------
# Animation: streaming response into a panel
# ---------------------------------------------------------------------------

def animate_response(skin, text: str, char_delay: float = 0.018) -> None:
    c       = skin.colors
    resp_bdr = c.get("response_border", "#888888")
    text_c  = c.get("banner_text",     "#dddddd")
    dim     = c.get("banner_dim",      "#555555")
    rlabel  = skin.branding.get("response_label", " Hermes ")

    displayed = ""
    with Live(console=console, refresh_per_second=30) as live:
        for char in text:
            displayed += char
            body = Text(displayed, style=text_c)
            live.update(Panel(body, border_style=resp_bdr,
                              title=f"[bold {resp_bdr}]{rlabel}[/]",
                              padding=(0, 1)))
            time.sleep(char_delay)


# ---------------------------------------------------------------------------
# Animation: typewriter for user prompt line
# ---------------------------------------------------------------------------

def typewriter(text: str, style: str = "", delay: float = 0.03) -> None:
    for char in text:
        console.print(char, end="", style=style)
        time.sleep(delay)
    console.print()


# ---------------------------------------------------------------------------
# Startup banner (static — shown before the animated conversation)
# ---------------------------------------------------------------------------

def render_banner(skin) -> None:
    c = skin.colors
    b = skin.branding

    border  = c.get("banner_border",  "#CD7F32")
    title_c = c.get("banner_title",   "#FFD700")
    accent  = c.get("banner_accent",  "#FFBF00")
    dim     = c.get("banner_dim",     "#B8860B")
    text_c  = c.get("banner_text",    "#FFF8DC")
    s_color = c.get("session_border", "#8B8682")
    agent   = b.get("agent_name",     "Hermes Agent")

    hero = skin.banner_hero if skin.banner_hero else _DEFAULT_HERO

    left_lines = ["", hero, ""]
    left_lines.append(
        f"[{accent}]{_MOCK_MODEL}[/]"
        f" [dim {dim}]·[/] [dim {dim}]{_MOCK_CTX}[/]"
        f" [dim {dim}]·[/] [dim {dim}]Nous Research[/]"
    )
    left_lines.append(f"[dim {dim}]{_MOCK_CWD}[/]")
    left_lines.append(f"[dim {s_color}]Session: {_MOCK_SESSION}[/]")

    right_lines = [f"[bold {accent}]Available Tools[/]"]
    for toolset, names in _MOCK_TOOLS.items():
        right_lines.append(
            f"[dim {dim}]{toolset}:[/] "
            + ", ".join(f"[{text_c}]{n}[/]" for n in names)
        )
    right_lines += ["", f"[bold {accent}]Available Skills[/]"]
    for cat, names in _MOCK_SKILLS.items():
        right_lines.append(f"[dim {dim}]{cat}:[/] [{text_c}]{', '.join(names)}[/]")

    total_tools  = sum(len(v) for v in _MOCK_TOOLS.values())
    total_skills = sum(len(v) for v in _MOCK_SKILLS.values())
    right_lines += [
        "",
        f"[dim {dim}]{total_tools} tools · {total_skills} skills · /help for commands[/]",
    ]

    grid = Table.grid(padding=(0, 2))
    grid.add_column("left",  justify="center")
    grid.add_column("right", justify="left")
    grid.add_row("\n".join(left_lines), "\n".join(right_lines))

    console.print(Panel(
        grid,
        title=f"[bold {title_c}]{agent} {_MOCK_VERSION}[/]",
        border_style=border,
        padding=(0, 2),
    ))


# ---------------------------------------------------------------------------
# Static detail sections (colors / spinner info / branding / status bar)
# ---------------------------------------------------------------------------

def render_detail(skin) -> None:
    c = skin.colors
    b = skin.branding

    title_c  = c.get("banner_title",    "#ffffff")
    accent   = c.get("banner_accent",   "#aaaaaa")
    dim      = c.get("banner_dim",      "#555555")
    text_c   = c.get("banner_text",     "#dddddd")
    resp_bdr = c.get("response_border", "#888888")
    st_bg    = c.get("status_bar_bg",   "#111111")
    st_str   = c.get("status_bar_strong", title_c)
    st_dim   = c.get("status_bar_dim",  dim)
    st_good  = c.get("status_bar_good", "#4caf50")
    ui_label = c.get("ui_label",        accent)

    sym      = b.get("prompt_symbol",  "❯")
    rlabel   = b.get("response_label", " Hermes ")
    welcome  = b.get("welcome",        "")
    goodbye  = b.get("goodbye",        "")
    help_hdr = b.get("help_header",    "")
    prefix   = skin.tool_prefix
    agent    = b.get("agent_name",     "Hermes Agent")

    # Colors
    section("Color Palette", accent)
    color_row(c, "banner_border",     "banner_border",     "banner_title",      "banner_title")
    color_row(c, "banner_accent",     "banner_accent",     "banner_dim",        "banner_dim")
    color_row(c, "banner_text",       "banner_text",       "ui_accent",         "ui_accent")
    color_row(c, "ui_label",          "ui_label",          "response_border",   "response_border")
    color_row(c, "prompt",            "prompt",            "input_rule",        "input_rule")
    color_row(c, "status_bar_bg",     "status_bar_bg",     "status_bar_text",   "status_bar_text")
    color_row(c, "status_bar_strong", "status_bar_strong", "status_bar_dim",    "status_bar_dim")
    color_row(c, "status_bar_good",   "status_bar_good",   "status_bar_warn",   "status_bar_warn")
    color_row(c, "status_bar_bad",    "status_bar_bad",    "status_bar_critical","status_bar_critical")
    color_row(c, "ui_ok",             "ui_ok",             "ui_error",          "ui_error")
    color_row(c, "session_label",     "session_label",     "session_border",    "session_border")

    # Spinner
    section("Spinner", accent)
    sp       = skin.spinner
    waiting  = sp.get("waiting_faces",  _DEFAULT_THINKING_FACES[:4])
    thinking = sp.get("thinking_faces", _DEFAULT_THINKING_FACES[:4])
    verbs    = sp.get("thinking_verbs", _DEFAULT_THINKING_VERBS)
    wings    = skin.get_spinner_wings()

    console.print("  " + "  ".join(f"[{title_c}]{f}[/]" for f in waiting[:6]),
                  highlight=False)
    console.print("  " + "  ".join(f"[{title_c}]{f}[/]" for f in thinking[:6]),
                  highlight=False)
    console.print(f"  [{dim}]" + " · ".join(verbs[:6]) + "[/]")
    if wings:
        lw, rw = wings[0]
        console.print(f"  [{title_c}]{lw} ⠹ {verbs[0]} {rw} (1.2s)[/]")

    # Branding
    section("Branding", accent)
    console.print(f"  [{ui_label}]Prompt:[/]      [{title_c}]{sym}[/]  ← input symbol")
    console.print(f"  [{ui_label}]Response:[/]    [{resp_bdr}]{rlabel}[/]  ← box header")
    console.print(f"  [{ui_label}]Tool prefix:[/] [{accent}]{prefix}[/]  ← tool output lines")
    console.print(f"  [{ui_label}]Welcome:[/]     [{dim}]{welcome}[/]")
    console.print(f"  [{ui_label}]Goodbye:[/]     [{dim}]{goodbye}[/]")
    console.print(f"  [{ui_label}]Help header:[/] [{dim}]{help_hdr}[/]")

    # Status bar
    section("Status Bar", accent)
    st = Text()
    st.append(f" {agent} ",      style=f"bold {st_str} on {st_bg}")
    st.append(" · ",             style=f"{st_dim} on {st_bg}")
    st.append(f"{_MOCK_MODEL} ", style=f"{st_str} on {st_bg}")
    st.append(" · ",             style=f"{st_dim} on {st_bg}")
    st.append("ctx 12% ",        style=f"{st_good} bold on {st_bg}")
    st.append(" · ",             style=f"{st_dim} on {st_bg}")
    st.append("$0.0014 ",        style=f"{st_str} on {st_bg}")
    st.append(" · ",             style=f"{st_dim} on {st_bg}")
    st.append("42s ",            style=f"{st_dim} on {st_bg}")
    console.print("  ", end="")
    console.print(st)
    console.print()


# ---------------------------------------------------------------------------
# Top-level: full preview for one skin
# ---------------------------------------------------------------------------

def preview_skin(name: str, data: dict, animate: bool = True) -> None:
    skin    = _build_skin_config(data)
    c       = skin.colors
    b       = skin.branding
    title_c = c.get("banner_title", "#ffffff")
    dim     = c.get("banner_dim",   "#555555")
    accent  = c.get("banner_accent","#aaaaaa")
    sym     = b.get("prompt_symbol","❯")

    console.print()
    console.print(Rule(
        title=f"[bold {title_c}]skin: {name}[/]  [dim {dim}]{data.get('description', '')}[/]",
        style=dim,
    ))
    console.print()

    # ── Logo ─────────────────────────────────────────────────────────────────
    logo = skin.banner_logo if skin.banner_logo else _DEFAULT_LOGO
    if shutil.get_terminal_size().columns >= 95:
        if animate:
            animate_logo(logo)
        else:
            console.print(logo)
        console.print()

    # ── Startup banner ───────────────────────────────────────────────────────
    render_banner(skin)

    if not animate:
        console.print()
        console.print(Rule(f"[dim {dim}]detail[/]", style=dim))
        render_detail(skin)
        return

    # ── Animated conversation demo ───────────────────────────────────────────
    console.print()
    console.print(Rule(f"[dim {dim}]conversation demo[/]", style=dim))
    console.print()

    # User prompt (typewriter)
    console.print(f"  [{title_c}]{sym}[/] ", end="")
    typewriter(_MOCK_PROMPT, style=f"bold {title_c}", delay=0.03)
    console.print()

    # Spinner + activity feed
    animate_spinner(skin, duration=4.2)
    console.print()

    # Streaming response
    animate_response(skin, _MOCK_RESPONSE, char_delay=0.012)
    console.print()

    # ── Static detail ────────────────────────────────────────────────────────
    console.print(Rule(f"[dim {dim}]detail[/]", style=dim))
    render_detail(skin)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args    = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags   = {a for a in sys.argv[1:] if a.startswith("--")}
    animate = "--no-animate" not in flags

    if "--all" in flags:
        names = list(_BUILTIN_SKINS.keys())
    elif args:
        invalid = [n for n in args if n not in _BUILTIN_SKINS]
        if invalid:
            console.print(f"[red]Unknown skin(s): {', '.join(invalid)}[/]")
            console.print(f"[dim]Available: {', '.join(_BUILTIN_SKINS.keys())}[/]")
            sys.exit(1)
        names = args
    else:
        names = ["default"]

    for name in names:
        preview_skin(name, _BUILTIN_SKINS[name], animate=animate)

    if len(names) > 1:
        console.print(Rule(style="dim"))
        console.print(f"  [dim]Previewed {len(names)} skins: {', '.join(names)}[/]\n")


if __name__ == "__main__":
    main()
