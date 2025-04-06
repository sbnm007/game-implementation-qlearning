#!/usr/bin/env python3
import argparse
import os
from games import TicTacToe, Connect4
from agents import (
    agent_wrapper_minimax, 
    agent_wrapper_minimax_ab, 
    agent_wrapper_qlearning,
    agent_wrapper_default, 
    agent_wrapper_random
)
from utils import load_q_table, save_q_table, train_qlearning, evaluate_algorithms, play_once
from visualizations import Visualizer

def main():
    parser = argparse.ArgumentParser()
    
    # Common arguments
    parser.add_argument("--game", type=str, choices=["tic_tac_toe","connect_4"], 
                        default="tic_tac_toe", help="Which game to play")
    parser.add_argument("--qtable_x_file", type=str, default="qtable_x.pkl", 
                    help="File to save/load Q-table for X player")
    parser.add_argument("--qtable_o_file", type=str, default="qtable_o.pkl", 
                    help="File to save/load Q-table for O player")
    parser.add_argument("--viz_dir", type=str, default="visualizations",
                    help="Directory for visualization output")
    
    # Mode selection
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Training mode
    train_parser = subparsers.add_parser("train", help="Train a Q-learning model")
    train_parser.add_argument("--train_as", type=str, choices=["X", "O"], 
                           default="X", help="Train as player X or O")
    train_parser.add_argument("--episodes", type=int, default=10000, 
                            help="Number of episodes to train")
    train_parser.add_argument("--opponent", type=str, 
                            choices=["random", "default", "minimax", "minimax_ab"], 
                            default="random", help="Opponent to train against")
    train_parser.add_argument("--depth_limit", type=int, default=4,
                            help="Depth limit for minimax opponent (if used)")
    train_parser.add_argument("--epsilon", type=float, default=0.1, 
                            help="Exploration rate")
    train_parser.add_argument("--alpha", type=float, default=0.1, 
                            help="Learning rate")
    train_parser.add_argument("--gamma", type=float, default=0.9, 
                            help="Discount factor")
    
    # Evaluation mode
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate algorithms against each other")
    eval_parser.add_argument("--episodes", type=int, default=100, 
                           help="Number of episodes for evaluation")
    eval_parser.add_argument("--depth_limit", type=int, default=None, 
                           help="Depth limit for minimax (9 for Tic Tac Toe, 4-6 for Connect4)")
    eval_parser.add_argument("--output", type=str, default=None, 
                           help="Output CSV file for results")
    
    # Play mode (for single algorithm vs opponent)
    play_parser = subparsers.add_parser("play", help="Play one algorithm against another")
    play_parser.add_argument("--algorithm", type=str, 
                          choices=["minimax","minimax_ab","qlearning"], 
                          default="minimax", help="Algorithm for player X")
    play_parser.add_argument("--opponent", type=str, 
                          choices=["default","random","qlearning","minimax","minimax_ab"], 
                          default="default", help="Algorithm for player O")
    play_parser.add_argument("--episodes", type=int, default=100, 
                          help="Number of episodes to play")
    play_parser.add_argument("--depth_limit", type=int, default=None, 
                          help="Depth limit for minimax")
    
    # New visualize mode
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations from existing data")
    viz_parser.add_argument("--q_file", type=str, required=True,
                         help="Q-table file to visualize")
    viz_parser.add_argument("--player", type=str, choices=["X", "O"], default="X",
                         help="Which player's Q-table to visualize")
    
    args = parser.parse_args()
    
    # Game choice
    GameClass = TicTacToe if args.game == "tic_tac_toe" else Connect4
    
    # Set default depth limit if not provided
    if args.depth_limit is None:
        if args.game == "tic_tac_toe":
            args.depth_limit = 9  # Full depth for Tic Tac Toe
        else:
            args.depth_limit = 4  # Limited depth for Connect4
    
    # Create visualizer
    visualizer = Visualizer(output_dir=args.viz_dir)
    
    # Load separate Q-tables for X and O
    q_table_x = load_q_table(args.qtable_x_file)
    q_table_o = load_q_table(args.qtable_o_file)
    
    # Combined q_tables dict for evaluation
    q_tables = {"X": q_table_x, "O": q_table_o}

    # Training mode
    if args.mode == "train":
        print(f"Training Q-learning as player {args.train_as} for {args.episodes} episodes " + 
              f"against {args.opponent} opponent...")
        if args.train_as == "X":
            train_qlearning(GameClass, q_table_x, episodes=args.episodes, 
                        alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, 
                        opponent=args.opponent, q_player="X", depth_limit=args.depth_limit,
                        visualizer=visualizer)
            save_q_table(q_table_x, args.qtable_x_file)
        else:  # Train as O
            train_qlearning(GameClass, q_table_o, episodes=args.episodes,
                        alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
                        opponent=args.opponent, q_player="O", depth_limit=args.depth_limit,
                        visualizer=visualizer)
            save_q_table(q_table_o, args.qtable_o_file)
        
    elif args.mode == "evaluate":
        print(f"Evaluating all algorithms against each other for {args.game}...")
        print(f"Using depth limit of {args.depth_limit} for minimax algorithms")
        evaluate_algorithms(
            GameClass,
            q_tables,
            episodes=args.episodes,
            depth_limit=args.depth_limit,
            output_file=args.output,
            visualizer=visualizer
        )
    
    elif args.mode == "visualize":
        # Load Q-table
        q_table = load_q_table(args.q_file)
        
        # Visualize Q-table
        game_type = GameClass.__name__.lower()
        visualizer.visualize_q_table(q_table, game_type, args.player)

    elif args.mode == "play":
        # Select agents
        if args.algorithm == "minimax":
            agentX = agent_wrapper_minimax
        elif args.algorithm == "minimax_ab":
            agentX = agent_wrapper_minimax_ab
        else:  # qlearning
            agentX = agent_wrapper_qlearning
        
        if args.opponent == "default":
            agentO = agent_wrapper_default
        elif args.opponent == "random":
            agentO = agent_wrapper_random
        elif args.opponent == "qlearning":
            agentO = agent_wrapper_qlearning
        elif args.opponent == "minimax":
            agentO = agent_wrapper_minimax
        else:  # minimax_ab
            agentO = agent_wrapper_minimax_ab
        
        # Play games
        results = {"X": 0, "O": 0, "Draw": 0}
        for i in range(args.episodes):
            game = GameClass()
            winner, _ = play_once(game, agentX, agentO, q_tables, depth_limit=args.depth_limit)
            if winner in results:
                results[winner] += 1
            
            # Print progress periodically
            if (i+1) % max(1, args.episodes // 10) == 0:
                print(f"Progress: {i+1}/{args.episodes} games")
        
        # Print final results
        print(f"Results for {args.algorithm} (X) vs {args.opponent} (O):")
        print(f"  X wins: {results['X']}, O wins: {results['O']}, Draws: {results['Draw']}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()