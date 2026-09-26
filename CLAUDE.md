# Algorithmic Trading

Repository of scripts and notebooks on algorithmic trading: strategy research, backtesting, execution costs and the statistics of performance evaluation.

## Organization

- Read the main README and the documentation of the folder you are working in.
- Place each project in `scripts/<project_name>/` with its own README; use descriptive snake_case names.
- Keep notebooks in `notebooks/`, tests in `tests/`, and small synthetic or publicly redistributable data in `data/examples/`.
- Write generated results to `outputs/`, creating the folder if it does not exist.
- Use paths relative to the repository root; never use absolute personal paths.
- Write documentation, code comments and commit messages in clear, professional English, and keep naming consistent with the existing code.

## Code and verification

- Respect the language of each project. Do not introduce frameworks or dependencies without a concrete need.
- With the first script, document the language version and create the appropriate dependency file if external packages are required.
- Document purpose, inputs, outputs and the exact run command using `docs/script-template.md`.
- State the strategy rules, signal timing, execution lag, transaction-cost assumptions, parameter values and the sources used.
- Prevent look-ahead bias: signals may use only information available before the trade, and tests must check it.
- Separate in-sample and out-of-sample periods, count every configuration tried, and report overfitting diagnostics where relevant.
- Fix and document the random seed of every simulation or bootstrap.
- Commit market data only when its license allows redistribution; otherwise, provide a download script and document the source.
- Run the relevant available checks and report the actual commands and outcomes. Never state that tests passed if they were not run.
- There is no general test command or CI workflow yet; each project documents its own checks.
- Update the catalogue in the main README whenever you add a script or notebook.

## Publishing

- Follow `docs/publishing.md` for branches, commits and pull requests.
- Review `git status` and the diff before committing, and stage only the files relevant to the change.
- Never commit API keys, tokens, passwords, personal data, confidential data or session transcripts.
- Use environment variables for credentials; any `.env.example` file must contain placeholder values only.
- Keep the existing MIT license and credit the sources of any reused code.
