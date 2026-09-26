# Learning notes

Teaching companion to the code in this repository. Each note explains one module:

1. intuition;
2. a numerical example;
3. the formulation, with every symbol defined;
4. derivations;
5. assumptions;
6. the mapping to functions and tests;
7. the choice between Python and C++;
8. statistical and financial interpretation;
9. failure modes and alternatives;
10. interview questions with answers;
11. exercises at four levels (answers not provided), with references to the original sources.

The notes are cumulative: later notes assume the conventions of earlier ones.

## Systematic-trading track

| Step | Note | Module | Status |
| --- | --- | --- | --- |
| 1 | [Futures, rolls and carry: from contract prices to tradable returns](futures/futures_rolls_and_carry.md) | `backtest_engine.futures`, `backtest_engine.schwartz_smith` | available |
| 2 | [Data snooping and multiple testing: inference after a search](statistics/data_snooping_and_multiple_testing.md) | `backtest_engine.validation`, `backtest_engine.cross_validation`, C++ joint bootstrap | available |
| 3 | Portfolio construction and risk: covariance estimation, risk parity, constraints, transaction costs | planned | planned |
| 4 | Execution and market impact: TWAP/VWAP, implementation shortfall, Almgren-Chriss | planned | planned |

## How to use a note

- **First read:** sections 1-3, then run the tests the note cites with `python -m pytest tests/backtest_engine -k <name>` and read them next to the derivations.
- **Interview preparation:** answer the questions in section 12 aloud before reading the answers, then attempt the level 2 and 3 exercises.
- **Research use:** section 10 (failure modes) is a checklist to apply to any backtest that uses the module.
