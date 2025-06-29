# ....................................Name: Fuseini Salifu............................................
def requirement1():
    board = [['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['1', '2', '3', '4', '5', '6', '7']]

    for row_1 in range(len(board)):
        for col_1 in range(len(board[0])):
            if board[row_1][col_1] == '0':
                board[row_1][col_1] = '.'

    board[5][0] = 'o'
    board[5][1] = 'x'
    board[5][2] = 'o'
    board[5][3] = 'x'
    board[4][3] = 'o'
    board[5][6] = 'x'

    for i in board:
        print(*i)


def board_reset():
    board = [['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['1', '2', '3', '4', '5', '6', '7']]

    for row_1 in range(len(board)):
        for col_1 in range(len(board[0])):
            if board[row_1][col_1] == '0':
                board[row_1][col_1] = '.'

    for i in board:
        print(*i)


def main():
    global game_over
    game_over = False

    board = [['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['0', '0', '0', '0', '0', '0', '0'],
             ['1', '2', '3', '4', '5', '6', '7']]

    for row_1 in range(len(board)):
        for col_1 in range(len(board[0])):
            if board[row_1][col_1] == '0':
                board[row_1][col_1] = '.'

    def looping():
        for i in board:
            print(*i)

    def check_winner():
        global game_over

        # Horizontal check
        for new_row in range(len(board)):
            for j in range(new_row):
                if (board[new_row][j] != '.' and board[new_row][j] == board[new_row][j + 1] == board[new_row][
                    j + 2]
                        == board[new_row][j + 3]):
                    board[new_row][j] = board[new_row][j].upper()
                    board[new_row][j + 1] = board[new_row][j + 1].upper()
                    board[new_row][j + 2] = board[new_row][j + 2].upper()
                    board[new_row][j + 3] = board[new_row][j + 3].upper()
                    print('CONNECT FOUR!!')
                    print(player1, 'Wins!'.upper())
                    game_over = True
                    looping()
                    return

        # Vertical check
        for new_col in range(4):
            for c in range(len(board[0])):
                if (board[new_col][c] != '.' and board[new_col][c] == board[new_col + 1][c] ==
                        board[new_col + 2][
                            c]
                        == board[new_col + 3][c]):
                    board[new_col][c] = board[new_col][c].upper()
                    board[new_col + 1][c] = board[new_col + 1][c].upper()
                    board[new_col + 2][c] = board[new_col + 2][c].upper()
                    board[new_col + 3][c] = board[new_col + 3][c].upper()
                    print('CONNECT FOUR!!')
                    print(player1, 'Wins!'.upper())
                    game_over = True
                    looping()
                    return

        # Diagonal from bottom-left to top-right
        for rows in range(3, 7):
            for cols in range(4):
                if (board[rows][cols] != '.' and board[rows][cols] == board[rows - 1][cols + 1] ==
                        board[rows - 2][cols + 2] == board[rows - 3][cols + 3]):
                    board[rows][cols] = board[rows][cols].upper()
                    board[rows - 1][cols + 1] = board[rows - 1][cols + 1].upper()
                    board[rows - 2][cols + 2] = board[rows - 2][cols + 2].upper()
                    board[rows - 3][cols + 3] = board[rows - 3][cols + 3].upper()
                    print('CONNECT FOUR')
                    print(player1, 'WINS')
                    game_over = True
                    looping()
                    return

        # Diagonal from bottom-right to Top-left
        for rows1 in range(4):
            for cols1 in range(4):
                if (board[rows1][cols1] != '.' and board[rows1][cols1] == board[rows1 + 1][cols1 + 1] ==
                        board[rows1 + 2][cols1 + 2] == board[rows1 + 3][cols1 + 3]):
                    board[rows1][cols1] = board[rows1][cols1].upper()
                    board[rows1 + 1][cols1 + 1] = board[rows1 + 1][cols1 + 1].upper()
                    board[rows1 + 2][cols1 + 2] = board[rows1 + 2][cols1 + 2].upper()
                    board[rows1 + 3][cols1 + 3] = board[rows1 + 3][cols1 + 3].upper()
                    print('CONNECT FOUR')
                    print(player1, 'WINS')
                    game_over = True
                    looping()
                    return

        if all(board[row2][col2] != '.' for row2 in range(len(board)) for col2 in range(len(board[0]))):
            print('DRAW')
            game_over = True
            return

    player1 = 'X'
    print('X to play')
    while not game_over:

        int_option = input('Enter an integer from 1-7: ')

        while type(int_option) == str:
            try:
                int_option = int(int_option)
                col_index = int_option - 1

                if 1 <= int_option <= 7:
                    if board[0][col_index] != '.':
                        print('Column full')
                        int_option = input('Enter an integer from 1-7: ')
                        continue

                    if int_option == 1:
                        for k in range(6, -1, -1):
                            if board[k][0] == '.':
                                if player1 == 'X':
                                    board[k][0] = 'x'
                                else:
                                    board[k][0] = 'o'
                                break
                        looping()
                    elif int_option == 2:
                        for k in range(6, -1, -1):
                            if board[k][1] == '.':
                                if player1 == 'X':
                                    board[k][1] = 'x'
                                else:
                                    board[k][1] = 'o'
                                break

                        looping()
                    elif int_option == 3:
                        for k in range(6, -1, -1):
                            if board[k][2] == '.':
                                if player1 == 'X':
                                    board[k][2] = 'x'
                                else:
                                    board[k][2] = 'o'
                                break
                        looping()
                    elif int_option == 4:
                        for k in range(6, -1, -1):
                            if board[k][3] == '.':
                                if player1 == 'X':
                                    board[k][3] = 'x'
                                else:
                                    board[k][3] = 'o'
                                break
                        looping()
                    elif int_option == 5:
                        for k in range(6, -1, -1):
                            if board[k][4] == '.':
                                if player1 == 'X':
                                    board[k][4] = 'x'
                                else:
                                    board[k][4] = 'o'
                                break
                        looping()
                    elif int_option == 6:
                        for k in range(6, -1, -1):
                            if board[k][5] == '.':
                                if player1 == 'X':
                                    board[k][5] = 'x'
                                else:
                                    board[k][5] = 'o'
                                break
                        looping()
                    else:
                        for k in range(6, -1, -1):
                            if board[k][6] == '.':
                                if player1 == 'X':
                                    board[k][6] = 'x'
                                else:
                                    board[k][6] = 'o'
                                break
                        looping()
                    check_winner()

                if int_option < 1 or int_option > 7:
                    print('Error, enter an integer from 1-7')
                    int_option = input('Enter an integer from 1-7: ')
            except ValueError:
                print('Error, enter an integer from 1-7')
                int_option = input('Enter an integer from 1-7: ')

        if game_over:
            break

        if player1 == 'X':
            player1 = 'O'
            print('O to play')
        else:
            player1 = 'X'
            print('X to play')


requirement1()
print()
print('Game Starting Now...')
print('Get ready')
print()
board_reset()
main()

# Ask the players if the want to play again
while True:
    play_again = input('Play again? y/n: ')
    if play_again == 'y':
        board_reset()
        main()
    elif play_again == 'n':
        print()
        print('Thank you for playing.')
        print('Goodbye!')
        break
    else:
        print('Please enter y or n')
