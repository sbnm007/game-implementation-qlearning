import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import datetime

class Visualizer:
    def __init__(self, output_dir="visualizations"):
        # Create visualization directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp-based subdirectory for current run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Set default style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def save_training_progress(self, episode_data, game_type, q_player, opponent_type):
        """
        Visualize training progress (wins, losses, draws over episodes)
        
        Parameters:
        - episode_data: Dict with 'episodes', 'wins', 'losses', 'draws', 'q_values'
        - game_type: String ("tic_tac_toe" or "connect_4")
        - q_player: String ("X" or "O")
        - opponent_type: String (opponent agent type)
        """
        plt.figure(figsize=(12, 8))
        
        # Plot wins, losses, draws
        plt.subplot(2, 1, 1)
        plt.plot(episode_data['episodes'], episode_data['wins'], 'g-', label='Wins')
        plt.plot(episode_data['episodes'], episode_data['losses'], 'r-', label='Losses')
        plt.plot(episode_data['episodes'], episode_data['draws'], 'b-', label='Draws')
        plt.xlabel('Episodes')
        plt.ylabel('Count')
        plt.title(f'Training Progress: {game_type.replace("_", " ").title()}, Q-Agent ({q_player}) vs {opponent_type}')
        plt.legend()
        plt.grid(True)
        
        # Plot win rate
        plt.subplot(2, 1, 2)
        total_games = np.array(episode_data['wins']) + np.array(episode_data['losses']) + np.array(episode_data['draws'])
        win_rate = np.array(episode_data['wins']) / np.where(total_games != 0, total_games, 1)
        plt.plot(episode_data['episodes'], win_rate * 100, 'g-')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate During Training')
        plt.grid(True)
        
        plt.tight_layout()
        filename = f"{self.run_dir}/training_{game_type}_{q_player}_vs_{opponent_type}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Training progress visualization saved to {filename}")
        
        # Q-value distribution
        if 'q_values' in episode_data and episode_data['q_values']:
            plt.figure(figsize=(10, 6))
            plt.hist(episode_data['q_values'], bins=50, alpha=0.75)
            plt.xlabel('Q-Value')
            plt.ylabel('Frequency')
            plt.title(f'Q-Value Distribution After Training')
            plt.grid(True)
            
            filename = f"{self.run_dir}/q_distribution_{game_type}_{q_player}_vs_{opponent_type}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Q-value distribution saved to {filename}")
    
    def visualize_q_table(self, q_table, game_type, player):
        """
        Create a visualization of the Q-table structure and value distributions
        """
        if not q_table:
            print("Q-table is empty, skipping visualization")
            return
            
        # Extract Q-values
        q_values = list(q_table.values())
        
        # Create a histogram of Q-values
        plt.figure(figsize=(10, 6))
        plt.hist(q_values, bins=50, color='skyblue', edgecolor='black')
        plt.title(f'Q-Values Distribution for {player} in {game_type}')
        plt.xlabel('Q-Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add mean and median lines
        mean_val = np.mean(q_values)
        median_val = np.median(q_values)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
        plt.legend()
        
        filename = f"{self.run_dir}/q_table_dist_{game_type}_{player}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Q-table distribution saved to {filename}")
    
    def save_evaluation_results(self, results, game_type):
        """
        Visualize evaluation results between different agents
        
        Parameters:
        - results: Dict with match results
        - game_type: String ("tic_tac_toe" or "connect_4")
        """
        # Create dataframe from results
        data = []
        for match, stats in results.items():
            if '(X) vs' in match:
                x_agent, o_agent = match.split(' (X) vs ')
                o_agent = o_agent.replace(' (O)', '')
                
                data.append({
                    'X_Agent': x_agent,
                    'O_Agent': o_agent,
                    'X_wins': stats['X_wins'],
                    'O_wins': stats['O_wins'],
                    'Draws': stats['Draws'],
                    'Avg_moves': stats['avg_moves'],
                    'Time': stats['time']
                })
        
        df = pd.DataFrame(data)
        
        # 1. Create a win rate comparison chart
        plt.figure(figsize=(12, 10))
        
        # Pivot data for visualization
        heatmap_data = df.pivot(index='X_Agent', columns='O_Agent', values='X_wins')
        
        # Create heatmap
        ax = plt.subplot(2, 2, 1)
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='g', cbar_kws={'label': 'X Wins'})
        plt.title('X Player Win Count')
        
        # Create draws heatmap
        ax = plt.subplot(2, 2, 2)
        heatmap_draws = df.pivot(index='X_Agent', columns='O_Agent', values='Draws')
        sns.heatmap(heatmap_draws, annot=True, cmap='PuRd', fmt='g', cbar_kws={'label': 'Draws'})
        plt.title('Draw Count')
        
        # Create average moves heatmap
        ax = plt.subplot(2, 2, 3)
        heatmap_moves = df.pivot(index='X_Agent', columns='O_Agent', values='Avg_moves')
        sns.heatmap(heatmap_moves, annot=True, cmap='Greens', fmt='.1f', cbar_kws={'label': 'Average Moves'})
        plt.title('Average Game Length (Moves)')
        
        # Create execution time heatmap
        ax = plt.subplot(2, 2, 4)
        heatmap_time = df.pivot(index='X_Agent', columns='O_Agent', values='Time')
        sns.heatmap(heatmap_time, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Execution Time (s)'})
        plt.title('Execution Time')
        
        plt.tight_layout()
        filename = f"{self.run_dir}/evaluation_heatmaps_{game_type}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Evaluation heatmaps saved to {filename}")
        
        # 2. Create a bar chart comparison
        plt.figure(figsize=(15, 8))
        
        # Add agent combinations as labels
        labels = [f"{row.X_Agent} vs {row.O_Agent}" for _, row in df.iterrows()]
        
        # Plot stacked bars
        x = np.arange(len(labels))
        width = 0.8
        
        plt.bar(x, df['X_wins'], width, label='X Wins', color='skyblue')
        plt.bar(x, df['O_wins'], width, bottom=df['X_wins'], label='O Wins', color='salmon')
        plt.bar(x, df['Draws'], width, bottom=df['X_wins'] + df['O_wins'], label='Draws', color='lightgreen')
        
        plt.xlabel('Agent Matchups')
        plt.ylabel('Game Outcomes')
        plt.title(f'Game Outcome Distribution - {game_type.replace("_", " ").title()}')
        plt.xticks(x, labels, rotation=90)
        plt.legend()
        
        plt.tight_layout()
        filename = f"{self.run_dir}/evaluation_bars_{game_type}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Evaluation bar chart saved to {filename}")
        
        # 3. Create a violin plot for move distribution where data is available
        if 'move_history' in results:
            plt.figure(figsize=(12, 8))
            move_data = []
            
            for match, moves in results['move_history'].items():
                for count in moves:
                    move_data.append({
                        'Match': match,
                        'Moves': count
                    })
            
            if move_data:
                moves_df = pd.DataFrame(move_data)
                sns.violinplot(x='Match', y='Moves', data=moves_df)
                plt.title(f'Distribution of Game Lengths - {game_type.replace("_", " ").title()}')
                plt.xticks(rotation=90)
                plt.tight_layout()
                
                filename = f"{self.run_dir}/moves_distribution_{game_type}.png"
                plt.savefig(filename)
                plt.close()
                print(f"Move distribution visualization saved to {filename}")

    def visualize_game_state(self, game, move_history=None, filename=None):
        """
        Visualize the current state of a game
        
        Parameters:
        - game: Game object (TicTacToe or Connect4)
        - move_history: List of moves made
        - filename: Output filename (optional)
        """
        if isinstance(game.__class__.__name__, str) and game.__class__.__name__ == "TicTacToe":
            self._visualize_tictactoe(game, move_history, filename)
        else:
            self._visualize_connect4(game, move_history, filename)
    
    def _visualize_tictactoe(self, game, move_history=None, filename=None):
        """Visualize a Tic-Tac-Toe game"""
        plt.figure(figsize=(6, 6))
        
        # Draw grid
        plt.plot([1, 1], [0, 3], 'k-')
        plt.plot([2, 2], [0, 3], 'k-')
        plt.plot([0, 3], [1, 1], 'k-')
        plt.plot([0, 3], [2, 2], 'k-')
        
        # Draw X's and O's
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                if game.board[idx] == 'X':
                    plt.plot([j+0.2, j+0.8], [2-i+0.2, 2-i+0.8], 'r-', linewidth=2)
                    plt.plot([j+0.8, j+0.2], [2-i+0.2, 2-i+0.8], 'r-', linewidth=2)
                elif game.board[idx] == 'O':
                    circle = plt.Circle((j+0.5, 2-i+0.5), 0.3, fill=False, ec='b', linewidth=2)
                    plt.gca().add_patch(circle)
        
        plt.xlim(-0.1, 3.1)
        plt.ylim(-0.1, 3.1)
        plt.title(f"Tic-Tac-Toe - Current Player: {game.current_player}")
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        else:
            filename = f"{self.run_dir}/tictactoe_state.png"
            plt.savefig(filename)
        
        plt.close()
        return filename
    
    def _visualize_connect4(self, game, move_history=None, filename=None):
        """Visualize a Connect4 game"""
        plt.figure(figsize=(8, 7))
        
        # Draw grid
        for i in range(game.rows + 1):
            plt.plot([0, game.cols], [i, i], 'b-')
        
        for j in range(game.cols + 1):
            plt.plot([j, j], [0, game.rows], 'b-')
        
        # Draw pieces
        for i in range(game.rows):
            for j in range(game.cols):
                if game.board[i][j] == 'X':
                    circle = plt.Circle((j+0.5, game.rows-i-0.5), 0.4, fc='r', ec='k')
                    plt.gca().add_patch(circle)
                elif game.board[i][j] == 'O':
                    circle = plt.Circle((j+0.5, game.rows-i-0.5), 0.4, fc='y', ec='k')
                    plt.gca().add_patch(circle)
        
        plt.xlim(-0.1, game.cols+0.1)
        plt.ylim(-0.1, game.rows+0.1)
        plt.title(f"Connect 4 - Current Player: {game.current_player}")
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        else:
            filename = f"{self.run_dir}/connect4_state.png"
            plt.savefig(filename)
        
        plt.close()
        return filename