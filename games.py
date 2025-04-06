# Game class implementations for Tic Tac Toe and Connect 4

class TicTacToe:
    def __init__(self):
        self.board = [" "] * 9  # 3x3 flattened
        self.current_player = "X"

    def copy(self):
        g = TicTacToe()
        g.board = self.board[:]
        g.current_player = self.current_player
        return g

    def get_legal_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == " "]

    def make_move(self, move):
        self.board[move] = self.current_player
        self.current_player = "O" if self.current_player == "X" else "X"

    def is_terminal(self):
        return self.check_winner() is not None or " " not in self.board

    def check_winner(self):
        wins = [(0,1,2),(3,4,5),(6,7,8),
                (0,3,6),(1,4,7),(2,5,8),
                (0,4,8),(2,4,6)]
        for a,b,c in wins:
            if self.board[a] != " " and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        if " " not in self.board:
            return "Draw"
        return None

    def print_board(self):
        for i in range(0, 9, 3):
            print("|" + "|".join(self.board[i:i+3]) + "|")
        print("-" * 9)  # Separator line

    def state_key(self):
        return "".join(self.board) + self.current_player


class Connect4:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = [[" " for _ in range(cols)] for __ in range(rows)]
        self.current_player = "X"

    def copy(self):
        g = Connect4(self.rows, self.cols)
        g.board = [row[:] for row in self.board]
        g.current_player = self.current_player
        return g

    def get_legal_moves(self):
        # Return columns that are not full
        return [c for c in range(self.cols) if self.board[0][c] == " "]

    def make_move(self, col):
        try:
            for r in range(self.rows - 1, -1, -1):
                if self.board[r][col] == " ":
                    self.board[r][col] = self.current_player
                    #print(f"Player {self.current_player} made move at row: {r}, col: {col}")  # Print the move
                    break
            else:
                raise ValueError("Column is full")  # Raise exception if column is full
        except ValueError as e:
            print(f"Error: Invalid move - {e}")
            raise  # Re-raise the exception to stop the game
        self.current_player = "O" if self.current_player == "X" else "X"

    def is_terminal(self):
        return self.check_winner() is not None or len(self.get_legal_moves()) == 0

    def check_winner(self):
        # Check horizontal, vertical, diagonal for 4 in a row
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if self.board[r][c] != " " and len(set([self.board[r][c+i] for i in range(4)])) == 1:
                    return self.board[r][c]
        for r in range(self.rows - 3):
            for c in range(self.cols):
                piece = self.board[r][c]
                if piece != " " and all(self.board[r+i][c] == piece for i in range(4)):
                    return piece
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if self.board[r][c] != " " and all(self.board[r+i][c+i] == self.board[r][c] for i in range(4)):
                    return self.board[r][c]
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if self.board[r][c] != " " and all(self.board[r-i][c+i] == self.board[r][c] for i in range(4)):
                    return self.board[r][c]
        if len(self.get_legal_moves()) == 0:
            return "Draw"
        return None

    def state_key(self):
        return "".join("".join(row) for row in self.board) + self.current_player

    def print_board(self):
        for row in self.board:
            print("|" + "|".join(row) + "|")
        print("-" * (self.cols * 2 + 1))  # Separator line
        print(" " + " ".join(str(i) for i in range(self.cols)))  # Column numbers