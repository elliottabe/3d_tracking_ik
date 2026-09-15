# Bash Safety: Common Mistakes & Footguns

A reference for the shell commands that most often cause accidental data loss or surprising
behavior — with a focus on what to watch for when an **LLM is driving the terminal**.

## Why this matters with LLMs

An assistant running shell commands can go wrong in ways that are easy to miss:

- It runs commands **confidently** and **chains several steps**, so one bad step executes
  before anyone reviews it.
- It may **assume the working directory** — a path-relative `rm`/`mv` in the wrong place is
  destructive.
- It may use a **variable that is empty or unset**, turning `rm -rf "$DIR/"` into `rm -rf /`.
- It may **paste a command from the internet** (`curl … | bash`) that runs unseen code.
- The shell **expands globs and variables before the command runs**, so what executes is not
  always what you read.

None of these are exotic — they're the everyday footguns below. Read a destructive command the
way the shell will actually execute it, not the way it looks.

> **The golden rule:** before anything that deletes, overwrites, or forces, **preview it**
> (`ls`, `echo`, `--dry-run`, `-n`, `git status`) and make sure it's **reversible** (committed
> to git or backed up).

---

## Quick danger cheat-sheet

| Command / pattern | What it does | Footgun | Safer practice |
|---|---|---|---|
| `rm -rf <path>` | Recursively, forcibly delete | No trash/undo; an empty `$path` widens the blast radius | Check the path; drop `-f`; `ls` it first; commit/back up |
| `rm *` | Delete all non-hidden files in cwd | Shell expands `*` to everything; wrong `pwd` = wrong files | `pwd` + `ls *` first; scope the glob; use `--` |
| `> file` | Truncate/overwrite (create empty) | Wipes the file **even if the command fails** | `>>` to append; `set -o noclobber`; write a temp then `mv` |
| `mv a b` / `cp a b` | Move / copy | Overwrites `b` silently | `-i` (prompt) or `-n` (no-clobber); confirm target |
| `git reset --hard` | Reset tree to a commit | Permanently discards uncommitted work | `git stash` first; check `git status` |
| `git clean -fdx` | Delete untracked + ignored files | Nukes `.env`, data, build dirs you wanted | Dry-run `git clean -ndx` first |
| `git push --force` | Overwrite remote branch | Clobbers others' commits | `--force-with-lease`; push to a branch |
| `chmod -R 777 <dir>` | Recursive perms = everyone rwx | Security hole; recursion hits more than you think | Least privilege; target specific files |
| `sudo <cmd>` | Run as root | Bypasses every safeguard | Avoid unless required; read the full command |
| `curl … \| bash` | Run a remote script now | Executes code you haven't seen | Download, **read it**, then run |
| `find … -exec rm {} \;` | Delete every match | Over-broad matches; surprising paths | `-print` first; then `-delete` deliberately |
| unquoted `$VAR` | Expand a variable | Empty/spaces → word-split or `rm -rf /` | Quote: `"$VAR"`; guard: `${VAR:?set me}` |
| `dd if=… of=…` | Raw block copy | Wrong `of=` destroys a disk | Rarely needed; triple-check `of=` |

---

## The footguns in detail

Each entry: **what it does**, **the footgun**, a **bad example**, and the **safe practice**.

### 1. `rm` and recursive deletion

- **What it does:** `rm` removes files. `-r` recurses into directories; `-f` ("force")
  suppresses prompts and ignores missing files. There is **no recycle bin** — deletion is
  immediate and permanent.
- **The footgun:** an empty or unset variable collapses the path. Trailing slashes and stray
  spaces change the target. A filename that starts with `-` is read as an option.
- **Bad examples:**
  ```bash
  rm -rf "$BUILD_DIR/"      # if BUILD_DIR is unset/empty -> rm -rf "/"
  rm -rf build /tmp/cache   # stray space: deletes ./build AND /tmp/cache (intended one path)
  rm -rf -- *               # at least this is honest about deleting everything here
  ```
- **Safe practice:** guard variables (`rm -rf "${BUILD_DIR:?refusing to run on empty}"`); run
  `ls "$BUILD_DIR"` first; avoid `-f` unless you truly need it; prefer moving to a temp/trash
  dir over deleting; use `--` to stop option parsing before globs/paths.

### 2. Wildcards / globbing (`*`, `?`, `[...]`, `**`)

- **What it does:** the **shell** expands globs **before** the command runs. `rm *` becomes
  `rm file1 file2 file3 …`. The command never sees the `*`.
- **The footgun:** `*` does **not** match dotfiles, so `rm -rf *` can leave `.env`/`.git`
  behind (and conversely `.*` historically matches `.` and `..` — dangerous). A file literally
  named `-rf` makes `rm *` expand to `rm -rf …`. An unmatched glob may be passed through
  literally (unless `nullglob` is set).
- **Bad example:**
  ```bash
  cp * backup            # if 'backup' is a file, not a dir, this overwrites it
  rm *.txt               # in the wrong directory, deletes the wrong .txt files
  ```
- **Safe practice:** preview the expansion with `echo *.txt` or `ls *.txt` first; confirm
  `pwd`; use `--` before globs; be explicit rather than relying on `*` for destructive ops.

### 3. Redirection & overwrite (`>`, `>>`, `<`, `2>&1`)

- **What it does:** `>` redirects stdout to a file, **creating or truncating it**; `>>`
  appends; `<` feeds input; `2>&1` merges stderr into stdout.
- **The footgun:** `>` truncates the target **before the command runs and even if it fails**.
  Reading and writing the same file in one line destroys it.
- **Bad example:**
  ```bash
  sort data.txt > data.txt    # truncates data.txt to empty BEFORE sort reads it
  generate > important.log    # if 'generate' errors, important.log is already emptied
  ```
- **Safe practice:** append with `>>` when you mean to add; write to a temp file then `mv`
  into place; enable `set -o noclobber` so `>` refuses to overwrite (use `>|` to force).

### 4. Variable expansion & quoting

- **What it does:** the shell substitutes `$VAR`, then **word-splits** and **glob-expands** the
  result unless it's quoted.
- **The footgun:** unquoted variables split on spaces and expand globs; unset/empty variables
  vanish, changing the command's meaning.
- **Bad example:**
  ```bash
  FILE="my report.txt"
  rm $FILE          # runs: rm my report.txt  -> tries to delete 'my' and 'report.txt'
  rm -rf $DIR/*     # if DIR is unset -> rm -rf /*
  ```
- **Safe practice:** always quote — `rm "$FILE"`; guard required vars with `${DIR:?message}`;
  turn on `set -u` to error on unset variables; use arrays for lists of paths.

### 5. Command chaining (`;`, `&&`, `||`, `&`)

- **What it does:** `;` runs the next command regardless; `&&` runs it only if the previous
  **succeeded**; `||` only if it **failed**; `&` backgrounds the command.
- **The footgun:** with `;`, a failed first command doesn't stop the second.
- **Bad example:**
  ```bash
  cd /path/to/build ; rm -rf *   # if 'cd' fails, rm -rf * runs in your CURRENT directory
  ```
- **Safe practice:** chain dependent steps with `&&` (`cd /path/to/build && rm -rf *`); in
  scripts use `set -euo pipefail` so failures stop execution.

### 6. Move / copy / sync overwrites (`mv`, `cp`, `rsync`)

- **What it does:** `mv`/`cp` overwrite the destination **without asking**. `rsync --delete`
  makes the destination match the source by **deleting extra files** in the destination.
- **The footgun:** a wrong destination silently clobbers; `rsync --delete` with a wrong target
  erases real data.
- **Safe practice:** use `-i` (interactive) or `-n` (no-clobber) for `mv`/`cp`; always
  `rsync -n` (dry-run) first and double-check the **trailing slash** on the source.

### 7. Recursive permissions & ownership (`chmod -R`, `chown -R`)

- **What it does:** recursively change permissions/ownership of a whole tree.
- **The footgun:** pointed at the wrong root (or `/`), it can break a system or open a security
  hole. `chmod -R 777` makes everything world-writable/executable.
- **Safe practice:** change the narrowest set of files needed; avoid `777`; double-check the
  target path before adding `-R`.

### 8. Git footguns

- **What they do / what you lose:**
  - `git reset --hard` — discards **uncommitted** changes permanently.
  - `git clean -fdx` — deletes **untracked + ignored** files (envs, data, build output).
  - `git checkout .` / `git restore .` — overwrites local edits with the index/HEAD.
  - `git push --force` — overwrites the remote branch, **clobbering others' commits**.
- **Safe practice:** `git status` before destructive ops; `git stash` to set work aside; use
  the dry-run `git clean -ndx` first; prefer `git push --force-with-lease`; never force-push a
  shared branch like `main`.

### 9. Running untrusted / remote code (`curl … | bash`, `eval`)

- **What it does:** pipes a downloaded script straight into a shell, or evaluates a string as
  code — running it **immediately, unseen**.
- **The footgun:** you execute whatever the URL serves right now (which can change), with your
  privileges.
- **Safe practice:** download to a file, **read it**, then run it
  (`curl -fsSL URL -o install.sh` → review → `bash install.sh`). Avoid `eval` on data you don't
  fully control.

### 10. Privilege & system-level commands (`sudo`, `dd`, `mkfs`)

- **What they do:** `sudo` runs as root; `dd` does raw block-level copies; `mkfs` formats a
  filesystem.
- **The footgun:** as root there are no guardrails; a wrong `of=` on `dd` or wrong device on
  `mkfs` destroys a disk irreversibly.
- **Safe practice:** avoid `sudo` unless required and read the whole command first; for
  `dd`/`mkfs`, verify the target device (`lsblk`) and never guess.

> **Do NOT run** a "fork bomb" such as `:(){ :|:& };:` — it recursively spawns processes until
> the machine is unusable. It's shown only so you recognize and refuse it.

---

## Reviewing an LLM's bash command — a 30-second checklist

Before approving a command the assistant proposes, ask:

1. **Is it destructive?** Does it delete, overwrite, force (`-f`), or recurse (`-r`/`-R`)?
2. **Are variables safe?** Every `$VAR` quoted (`"$VAR"`) and guaranteed non-empty
   (`${VAR:?}`)?
3. **Is the location right?** Correct `pwd` and absolute/relative paths? No stray spaces?
4. **Was it previewed?** A dry-run / `ls` / `echo` / `git status` ran first?
5. **Is it reversible?** Work committed to git or backed up so a mistake can be undone?

If any answer is "not sure," stop and verify before running. When in doubt, prefer the
read-only or `--dry-run` version of the command first.
