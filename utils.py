import os
import pickle
import random
import time
import csv
import math
import numpy as np
from agents import minimax, default_agent, qlearning_move

def save_q_table(q_table, filename):
    with open(filename, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Q-table saved to {filename}")

def load_q_table(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            print(f"Loaded Q-table from {filename}")
            return pickle.load(f)
    print(f"No existing Q-table found at {filename}, creating new")
    return {}

def train_qlearning(game_class, q_table, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1,
                   opponent="random", q_player="X", depth_limit=4, visualizer=None):
    # Select opponent for training against
    if opponent == "default":
        opponent_func = default_agent
    elif opponent == "minimax":
        def minimax_opponent(game):
            val, move = minimax(game, depth=depth_limit)
            return move
        opponent_func = minimax_opponent
    elif opponent == "minimax_ab":
        def minimax_ab_opponent(game):
            val, move = minimax(game, depth=depth_limit, alpha=-math.inf, beta=math.inf)
            return move
        opponent_func = minimax_ab_opponent
    else:  # random
        opponent_func = lambda game: random.choice(game.get_legal_moves())

    # Progress tracking
    win_count = 0
    loss_count = 0
    draw_count = 0

    # Visualization data tracking
    track_interval = max(1, episodes // 100)  # Track at most 100 data points
    tracking_data = {
        'episodes': [],
        'wins': [],
        'losses': [],
        'draws': [],
        'q_values': []
    }

    for i in range(episodes):
        g = game_class()
        episode_win = False
        episode_loss = False
        episode_draw = False

        # Play one episode
        while not g.is_terminal():
            s = g.state_key()  # Get state before the move
            if g.current_player == q_player:  # Q-learning agent
                a = qlearning_move(g, q_table, epsilon)
                if a is None:
                    break

                g.make_move(a)

            else:
                move = opponent_func(g)
                if move is None:
                    break
                g.make_move(move)

            reward = 0
            if g.is_terminal():
                w = g.check_winner()
                if w == q_player:
                    reward = 1
                    win_count += 1
                    episode_win = True
                elif w not in [None, "Draw"]:
                    reward = -1
                    loss_count += 1
                    episode_loss = True
                elif w == "Draw":
                    draw_count += 1
                    episode_draw = True

            # Update Q
            if g.current_player != q_player and a is not None:
                old_q = q_table.get(s + str(a), 0)
                ns = g.state_key()  # Use the actual game state g
                legal_moves = g.get_legal_moves()  # Use the actual game state g
                future_q = max([q_table.get(ns + str(m), 0) for m in legal_moves] or [0]) if legal_moves else 0
                new_q = old_q + alpha * (reward + gamma * future_q - old_q)
                q_table[s + str(a)] = new_q

        # Track data for visualization
        if i % track_interval == 0 or i == episodes - 1:
            tracking_data['episodes'].append(i)
            tracking_data['wins'].append(win_count)
            tracking_data['losses'].append(loss_count)
            tracking_data['draws'].append(draw_count)

        # Report progress
        if (i + 1) % (episodes // 10) == 0:
            print(f"Training progress: {i + 1}/{episodes} episodes")
            print(f"Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count}")

    # Add final Q-values to tracking data
    tracking_data['q_values'] = list(q_table.values())

    # Generate visualizations if visualizer is provided
    if visualizer:
        game_type = game_class.__name__.lower()
        visualizer.save_training_progress(tracking_data, game_type, q_player, opponent)
        visualizer.visualize_q_table(q_table, game_type, q_player)

    print(f"Training complete. Size of Q-table: {len(q_table)} state-actions")
    return q_table


def play_once(game, agentX, agentO, q_tables=None, depth_limit=4):
    print(f"Evaluating game: {game}")  # Add this line

    moves_made = 0
    while not game.is_terminal():
        if game.current_player == "X":
            move = agentX(game, q_tables, depth=depth_limit)
            agent_name = agentX.__name__  # Get the agent's name
        else:
            move = agentO(game, q_tables, depth=depth_limit)
            agent_name = agentO.__name__  # Get the agent's name

        if move is None:
            print("Error: Agent returned None move")
            break

        print(f"Player {game.current_player} ({agent_name}) makes move: {move}")  # Print the move
        game.make_move(move)
        moves_made += 1

    winner = game.check_winner()
    return winner, moves_made

def evaluate_algorithms(game_class, q_tables, episodes=100, depth_limit=4, output_file=None, visualizer=None):
    print(f"Evaluating game: {game_class.__name__}")  # Add this line

    from agents import (
        agent_wrapper_minimax, 
        agent_wrapper_minimax_ab, 
        agent_wrapper_qlearning,
        agent_wrapper_default, 
        agent_wrapper_random
    )
    
    agents = {
        "minimax": agent_wrapper_minimax,
        "minimax_ab": agent_wrapper_minimax_ab,
        "qlearning": agent_wrapper_qlearning,
        "default": agent_wrapper_default,
        "random": agent_wrapper_random
    }
    
    results = {}
    move_history = {}  # For tracking move counts
    
    for x_name, x_agent in agents.items():
        for o_name, o_agent in agents.items():
            match_key = f"{x_name} (X) vs {o_name} (O)"
            print(f"Evaluating: {match_key}")
            
            results[match_key] = {
                "X_wins": 0,
                "O_wins": 0,
                "Draws": 0,
                "avg_moves": 0,
                "time": 0
            }
            
            move_history[match_key] = []
            total_moves = 0
            start_time = time.time()
            
            for i in range(episodes):
                game = game_class()
                winner, moves = play_once(game, x_agent, o_agent, q_tables, depth_limit)
                total_moves += moves
                move_history[match_key].append(moves)
                
                if winner == "X":
                    results[match_key]["X_wins"] += 1
                elif winner == "O":
                    results[match_key]["O_wins"] += 1
                elif winner == "Draw":
                    results[match_key]["Draws"] += 1
            
            results[match_key]["time"] = time.time() - start_time
            results[match_key]["avg_moves"] = total_moves / episodes
            
            print(f"  X wins: {results[match_key]['X_wins']}, " +
                  f"O wins: {results[match_key]['O_wins']}, " +
                  f"Draws: {results[match_key]['Draws']}, " +
                  f"Avg moves: {results[match_key]['avg_moves']:.1f}")
    
    # Add move history for visualization
    results['move_history'] = move_history
    
    # Save results to file if specified
    if output_file:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['Match', 'X_wins', 'O_wins', 'Draws', 'avg_moves', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for match, data in results.items():
                if match != 'move_history':  # Skip the move history when writing to CSV
                    row = {
                        'Match': match,
                        'X_wins': data['X_wins'],
                        'O_wins': data['O_wins'],
                        'Draws': data['Draws'],
                        'avg_moves': f"{data['avg_moves']:.1f}",
                        'time': f"{data['time']:.2f}s"
                    }
                    writer.writerow(row)
        
        print(f"Results saved to {output_file}")
    
    # Generate visualizations if visualizer is provided
    if visualizer:
        game_type = game_class.__name__.lower()
        visualizer.save_evaluation_results(results, game_type)
    
    return results