def math_calculation():
    # Ask user for inputs
    num1 = float(input('Enter first number: '))
    desired_operator = input('Enter operator(+, -, *, or /): ')
    num2 = float(input('Enter second number: '))

    # Perform calculations based on user inputs
    if desired_operator == '+':
        print('Answer:', (num1 + num2))
    elif desired_operator == '*':
        print('Answer:', (num1 * num2))
    elif desired_operator == '/':
        print('Answer:', (num1 / num2))
    elif desired_operator == '**':
        print('Answer:', (num1 ** num2))
    elif desired_operator == '-':
        print('Answer:', (num1 - num2))
    else:
        print('Invalid operator! Try again')

        # Continues to ask user for inputs if user enters a wrong operator or number

        while desired_operator != '+' or '-' or '*' or '/':
            num1 = float(input('Enter first number: '))
            desired_operator = input('Enter operator(+, -, *, or /): ')
            num2 = float(input('Enter second number: '))

            if desired_operator == '+':
                print('Answer:', (num1 + num2))
            elif desired_operator == '*':
                print('Answer:', (num1 * num2))
            elif desired_operator == '/':
                print('Answer:', (num1 / num2))
            elif desired_operator == '**':
                print('Answer:', (num1 ** num2))
            elif desired_operator == '-':
                print('Answer:', (num1 - num2))
            else:
                print('Invalid operator! Try again')

            option = input('Would you like to continue? y/n:')

            if option == 'y':
                math_calculation()
            else:
                verify = input('Are you sure? y/n: ')
                if verify == 'y':
                    print('Goodbye!')
                    break
                else:
                    math_calculation()


math_calculation()
