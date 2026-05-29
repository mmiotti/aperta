# Contributing to aperta

Internal repository for now — external contributions open after the
toolkit-paper publication. The setup notes below are relevant for
collaborators with commit access in the meantime.

## Editing notebooks

Notebooks here are **jupytext-paired**: each `.ipynb` has a `.py` shadow
under the same basename. Edits in either form propagate to the other via
`jupytext --sync <file>`. The `.py` shadow is the human-readable form for
code review; the `.ipynb` carries the executed outputs.

Two pieces of automation handle the friction. Both need to be activated
once per local clone:

```bash
# 1. Auto-sync .py <-> .ipynb on every commit (jupytext via pre-commit).
pip install pre-commit
pre-commit install

# 2. Strip outputs from non-example notebooks on commit
#    (keeps git history clean; example notebooks are exempt and ship
#    with their outputs intact so figures render on GitHub).
pip install nbstripout
nbstripout --install
```

After this setup:

- Edit either the `.py` or the `.ipynb` — the pre-commit hook keeps the
  pair in sync (if your commit changes only one side, the hook updates
  the other and asks you to re-stage).
- `git diff` and `git log` on non-example `.ipynb` files show only code
  and markdown changes (outputs stripped by `nbstripout`).
- Notebooks under [`examples/`](examples/) are exempt from output-stripping
  via [`.gitattributes`](.gitattributes) and ship with their executed
  outputs intact. Before committing changes to an example notebook,
  execute it end-to-end (in VSCode / Jupyter, or via
  `jupytext --sync --execute <file>`) so the committed outputs reflect
  the current code.

If you skip the `nbstripout --install` step, your commits to non-example
notebooks will include output cells (large diffs, slow GitHub renders).
If you skip the `pre-commit install` step, you'll need to run
`jupytext --sync` manually after notebook edits so the `.py` and `.ipynb`
don't drift apart. Both are one-time setups; install them.
