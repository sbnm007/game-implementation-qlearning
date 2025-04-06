import random
import math

# Core agent implementation
def minimax(game, depth=9, alpha=None, beta=None):
    # If terminal or at depth limit, return utility
    if game.is_terminal() or depth == 0:
        w = game.check_winner()
        if w == "X":
            return (1, None)
        elif w == "O":
            return (-1, None)
        elif w == "Draw":
            return (0, None)
        # Non-terminal but depth=0
        return (0, None)

    if game.current_player == "X":
        best_val = -math.inf
        best_move = None
        for move in game.get_legal_moves():
            child = game.copy()
            child.make_move(move)
            val, _ = minimax(child, depth-1, alpha, beta)
            if val > best_val:
                best_val = val
                best_move = move
            if alpha is not None:
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
        return (best_val, best_move)
    else:
        best_val = math.inf
        best_move = None
        for move in game.get_legal_moves():
            child = game.copy()
            child.make_move(move)
            val, _ = minimax(child, depth-1, alpha, beta)
            if val < best_val:
                best_val = val
                best_move = move
            if beta is not None:
                beta = min(beta, val)
                if beta <= alpha:
                    break
        return (best_val, best_move)

def default_agent(game):
    # If can win, do it. If need to block, do it. Else random.
    for move in game.get_legal_moves():
        c = game.copy()
        c.make_move(move)
        if c.check_winner() == game.current_player:
            return move
    # Block
    opp = "O" if game.current_player == "X" else "X"
    for move in game.get_legal_moves():
        c = game.copy()
        c.make_move(move)
        if c.check_winner() == opp:
            return move
    return random.choice(game.get_legal_moves()) if game.get_legal_moves() else None

def qlearning_move(game, q_table, epsilon=0.1):
    state = game.state_key()
    moves = game.get_legal_moves()
    if random.random() < epsilon:
        return random.choice(moves)
    
    # Choose best move from Q-table
    best_move = max(moves, key=lambda m: q_table.get(state + str(m), 0), default=None)
    
    # If no best move found (e.g., no Q-values for available moves), choose a random move
    return best_move if best_move is not None else random.choice(moves)

# Agent wrappers for consistency in the evaluate/play methods
def agent_wrapper_minimax(game, q_tables, depth=9):
    val, move = minimax(game, depth=depth)
    return move

def agent_wrapper_minimax_ab(game, q_tables, depth=9):
    val, move = minimax(game, depth=depth, alpha=-math.inf, beta=math.inf)
    return move

def agent_wrapper_default(game, q_tables, depth=9):
    return default_agent(game)

def agent_wrapper_random(game, q_tables, depth=9):
    moves = game.get_legal_moves()
    return random.choice(moves) if moves else None

def agent_wrapper_qlearning(game, q_tables, depth=9):
    # Use appropriate Q-table based on player
    q_table = q_tables["X"] if game.current_player == "X" else q_tables["O"]
    return qlearning_move(game, q_table, epsilon=0.0)