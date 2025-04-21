# /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_3_Evolution_of_Cooperation

# Adaptive Generous Tit-for-Tat Strategy

**Author**: BC  
**Course**: Evolution through Programming  
**Lecturer**: Prof. Yitzhak Pilpel  
**TA**: Omer Kerner – omer.kerner@weizmann.ac.il  
**Assignment**: 3 – Evolution of Cooperation

## Overview

This project implements an adaptive and robust strategy for the *Iterated Prisoner's Dilemma*, inspired by the *Generous Tit-for-Tat (GTFT)* approach. The strategy is designed to identify and respond dynamically to various opponent types, promoting mutual cooperation while resisting exploitation.

The strategy is defined in `src/strategy.py` and adheres strictly to the assignment specifications: it is self-contained, uses only built-in Python libraries, and maintains internal state solely through the provided history parameters.

## Strategy Description

The core idea behind the strategy is to **start cooperatively**, monitor the opponent’s behavior, and **adjust the forgiveness rate** dynamically. It weighs recent moves more heavily than older ones to react to changing tactics.

Key features:
- **Adaptive forgiveness**: Becomes more forgiving against cooperative opponents and stricter against defectors.
- **Defection recovery**: Detects mutual defection loops and increases the chance of breaking the cycle.
- **Opponent profiling**:
  - Against *Always Defect*: occasional testing for a change.
  - Against *Tit-for-Tat*: avoids long retaliation cycles.
  - Against *Random*: stabilizes toward cautious cooperation.
  - Against unknown strategies: adjusts based on recent history.

## Files

- `src/strategy.py`: Main strategy implementation (submit this file).
- `run_strategy.ipynb`: Jupyter notebook for running and visualizing simulations against test opponents.
- `README.md`: This documentation file.
- `requirements.txt`: Empty by default (no external dependencies).

## How to Run

You can simulate and visualize the strategy using the included Jupyter Notebook:

```bash
jupyter notebook src/run_strategy.ipynb
```

Or run simulations directly in Python by importing and calling the `strategy` function.

## Evaluation Support

The notebook tests the strategy against four benchmark opponents:
- Always Cooperate
- Always Defect
- Tit-for-Tat
- Random Player

Each simulation plots the moves of both players across 50 rounds, helping to visually analyze strategic behavior and adaptability.

