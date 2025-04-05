# Game AI Framework

This repository contains implementations of Tic-Tac-Toe and Connect 4 games with various AI agents: Minimax (with/without Alpha-Beta pruning), Q-learning, and simple heuristic agents.

## File Structure

- `main.py`: Command-line interface for training, evaluation, and gameplay
- `games.py`: Game class implementations (TicTacToe and Connect4)
- `agents.py`: AI agent implementations (Minimax, Q-learning, etc.)
- `utils.py`: Utility functions for training, evaluation, and file operations

## Usage

### Training a Q-learning agent

```bash
python main.py train --game tic_tac_toe --train_as X --episodes 10000 --opponent minimax
```

### Evaluating different algorithms

```bash
python main.py evaluate --game connect_4 --episodes 100 --output results.csv
```

### Playing games with specific algorithms

```bash
python main.py play --game tic_tac_toe --algorithm minimax_ab --opponent default --episodes 100
```

## Configuration Options

### Common Arguments
- `--game`: Choose between "tic_tac_toe" or "connect_4"
- `--qtable_x_file`: File to save/load Q-learning data for X player
- `--qtable_o_file`: File to save/load Q-learning data for O player

### Training Arguments
- `--train_as`: Train as player "X" or "O"
- `--episodes`: Number of training episodes
- `--opponent`: Opponent for training ("random", "default", "minimax", "minimax_ab")
- `--depth_limit`: Depth limit for minimax opponent
- `--epsilon`, `--alpha`, `--gamma`: Q-learning parameters

### Evaluation Arguments
- `--episodes`: Number of evaluation episodes
- `--depth_limit`: Depth limit for minimax algorithms
- `--output`: Output CSV file for results

### Play Arguments
- `--algorithm`: Algorithm for player X
- `--opponent`: Algorithm for player O
- `--episodes`: Number of games to play
- `--depth_limit`: Depth limit for minimax

## Algorithms

1. **Minimax**: Classical game tree search algorithm
2. **Minimax with Alpha-Beta Pruning**: Optimized minimax with pruning
3. **Q-learning**: Reinforcement learning approach
4. **Default Agent**: Simple heuristic (win if possible, block if necessary, otherwise random)
5. **Random Agent**: Makes random legal moves
