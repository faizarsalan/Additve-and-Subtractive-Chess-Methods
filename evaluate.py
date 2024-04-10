import chess
import numpy as np
import tensorflow as tf
import time

# Define mapping for each chess piece to its integer representation
mapping = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
}

# Convert FEN to board tensor
def fen_to_board_tensor(fen):
    board = chess.Board(fen)
    tensor = [[0 for _ in range(8)] for _ in range(8)]
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            tensor[7 - square // 8][square % 8] = mapping[piece.symbol()]
    return tensor

# Define the function to get the best move
def get_best_move(board):
    best_move = None
    best_delta = 0  # Initializing to 0

    is_white_turn = board.turn == chess.WHITE

    for move in board.legal_moves:
        # print(move)

        board.push(move)
        board_tensor = fen_to_board_tensor(board.fen())
        board_tensor_reshaped = np.array(board_tensor).reshape(1, 8, 8, 1)

        predicted_delta = model.predict(board_tensor_reshaped)[0][0]
        # print("Predicted Delta:", predicted_delta)

        if is_white_turn:
            if best_move is None or predicted_delta > best_delta:  
                best_delta = predicted_delta
                best_move = move

        else:
            if best_move is None or predicted_delta < best_delta:  
                best_delta = predicted_delta
                best_move = move

        board.pop()

    return best_move

# Test the function
fen_positions = ["1rbq1rk1/p1b1nppp/1p2p3/8/1B1pN3/P2B4/1P3PPP/2RQ1R1K w - - 0 1"
,"3r2k1/p2r1p1p/1p2p1p1/q4n2/3P4/PQ5P/1P1RNPP1/3R2K1 b - - 0 1"
,"3r2k1/1p3ppp/2pq4/p1n5/P6P/1P6/1PB2QP1/1K2R3 w - - 0 1"
,"r1b1r1k1/1ppn1p1p/3pnqp1/8/p1P1P3/5P2/PbNQNBPP/1R2RB1K w - - 0 1"
,"2r4k/pB4bp/1p4p1/6q1/1P1n4/2N5/P4PPP/2R1Q1K1 b - - 0 1"
,"r5k1/3n1ppp/1p6/3p1p2/3P1B2/r3P2P/PR3PP1/2R3K1 b - - 0 1"
,"2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - - 0 1"
,"5r1k/6pp/1n2Q3/4p3/8/7P/PP4PK/R1B1q3 b - - 0 1"
,"r3k2r/pbn2ppp/8/1P1pP3/P1qP4/5B2/3Q1PPP/R3K2R w KQkq - 0 1"
,"3r2k1/ppq2pp1/4p2p/3n3P/3N2P1/2P5/PP2QP2/K2R4 b - - 0 1"
,"q3rn1k/2QR4/pp2pp2/8/P1P5/1P4N1/6n1/6K1 w - - 0 1"
,"6k1/p3q2p/1nr3pB/8/3Q1P2/6P1/PP5P/3R2K1 b - - 0 1"
,"1r4k1/7p/5np1/3p3n/8/2NB4/7P/3N1RK1 w - - 0 1"
,"1r2r1k1/p4p1p/6pB/q7/8/3Q2P1/PbP2PKP/1R3R2 w - - 0 1"
,"r2q1r1k/pb3p1p/2n1p2Q/5p2/8/3B2N1/PP3PPP/R3R1K1 w - - 0 1"
,"8/4p3/p2p4/2pP4/2P1P3/1P4k1/1P1K4/8 w - - 0 1"
,"1r1q1rk1/p1p2pbp/2pp1np1/6B1/4P3/2NQ4/PPP2PPP/3R1RK1 w - - 0 1"
,"q4rk1/1n1Qbppp/2p5/1p2p3/1P2P3/2P4P/6P1/2B1NRK1 b - - 0 1"
,"r2q1r1k/1b1nN2p/pp3pp1/8/Q7/PP5P/1BP2RPN/7K w - - 0 1"
,"8/5p2/pk2p3/4P2p/2b1pP1P/P3P2B/8/7K w - - 0 1"
,"8/2k5/4p3/1nb2p2/2K5/8/6B1/8 w - - 0 1"
,"1B1b4/7K/1p6/1k6/8/8/8/8 w - - 0 1"
,"rn1q1rk1/1b2bppp/1pn1p3/p2pP3/3P4/P2BBN1P/1P1N1PP1/R2Q1RK1 b - - 0 1"
,"8/p1ppk1p1/2n2p2/8/4B3/2P1KPP1/1P5P/8 w - - 0 1"
,"8/3nk3/3pp3/1B6/8/3PPP2/4K3/8 w - - 0 1"
,"rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 1"
,"rnbqk2r/pppp1ppp/5n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
,"8/3P3k/n2K3p/2p3n1/1b4N1/2p1p1P1/8/3B4 w - - 0 1"]
output_moves = []
best_moves = ["e4f6"
,"f5d4"
,"h4h5"
,"b1b2"
,"g5c1"
,"h7h6"
,"b7e4"
,"h7h6"
,"f3e2"
,"d5c3"
,"g3f5"
,"c6d6"
,"c3d5"
,"b1b2"
,"d3f5"
,"b3b4"
,"e4e5"
,"a8c8"
,"a4d7"
,"h3g4"
,"c4b5"
,"b8a7"
,"b7a6"
,"e4c6"
,"b5d7"
,"d8h4"
,"h5f7"
,"g4f6"]

min_seed = 1
max_seed = 101
batch_size = 512
method_name = 'SUBTRACTIVE'

# Assuming fen_positions and best_moves are lists that you've defined somewhere:
results = {}

for seed in range(min_seed, max_seed):
    future_file = f'./gmonly_{batch_size}/{batch_size}seed_{seed}.h5'
    output_moves = []
    # Load the trained model
    model = tf.keras.models.load_model(future_file)
    for fen_position in fen_positions:
        board = chess.Board(fen_position)
        output_moves.append(get_best_move(board))

    correct_positions = [i for i, (best, out) in enumerate(zip(best_moves, output_moves), 1) if str(best) == str(out)]
    correct_count = len(correct_positions)
    results[seed] = correct_positions

best_seed = max(results, key=results.get)
best_correct_positions = results[best_seed]

save_path = f'{method_name}eval{batch_size}_{min_seed}_{max_seed}.txt'

# Save results to a file
with open(save_path, 'w') as file:
    for seed, positions in results.items():
        file.write(f"{seed}: {positions}\n")

print(f"Results saved to {save_path}")
import time
print(time.ctime())