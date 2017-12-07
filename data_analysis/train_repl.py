import train
import matplotlib.pyplot as plt


if __name__ == '__main__':
    print('press h for help')
    cont = 'y'

    while cont:
        # Read
        selection = input('train_repl>>> ')
        
        # Eval, Print
        if selection == 'h':
            print("""Type 'h' for help.
Type 'exit' to quit.
Type 'init tr' to initialize training environment.
Type 'init sess' to (re)initialize tensorflow session.
Type 'set' to set active data record.
Type 'train n' to perform n training iterations. 
Type 'status' to show a summary of current activity.
Type 'vis' to show the progress of training
Type 'alter to update the parameters of the model. This will reinitialize the session.
""")
        # Quit training session
        elif selection == 'quit' or selection == 'exit':
            print('goodbye')
            cont = ''

        # Initialize or reinitialize training components
        elif selection.startswith('init'):
            if selection.split(' ')[1].startswith('tr'):
                record_num = input('Set active record: ')
                ml = train.TrainNN(record_num)
                a = []
                b = []

            elif selection.split(' ')[1].startswith('sess'):
                try:
                    ml
                    ml.reinit_tf_session()
                    a = []
                    b = []

                except NameError:
                    print('Initialize training object before session')

            else:
                print('Command not found. Press h for help')

        elif selection.startswith('set'):
            record_num = input('Enter active record number: ')
            ml.set_active_record(record_num)

        elif selection.startswith('status'):
            print('\n')
            ml.show_params()
            print('\n')

        elif selection.startswith('train'):
            try:
                n = int(selection.split(' ')[1])
                a_plus, b_plus = ml.sgd(n)
                a = a + a_plus
                b = b + b_plus
            except ValueError:
                print('number of iterations must be a number')

        elif selection.startswith('vis'):
            try:
                plt.plot(a)
                plt.plot(b)
                plt.title('Error per iteration')
                plt.show()
            except NameError:
                print('Initialize training object before visualizing')

        elif selection.startswith('alter'):
            try:
                record_num = ml.active_record
                del ml

                ml = train.TrainNN(record_num)

                par = input("""
    Enter param number to alter:
    1) learning rate
    2) regularization rate
    3) parameter size (stddev)
    4) hidden units\n""")
                if par == '1':
                    ml.learning_rate = float(input('Enter new value: '))
                elif par == '2':
                    ml.reg_rate = float(input('Enter new value: '))
                elif par == '3':
                    ml.param_size = float(input('Enter new value: '))
                elif par == '4':
                    ml.num_hidden_1 = int(input('Enter new value: '))
                else:
                    print('Parameter option not found')

                ml.reinit_tf_session()
                a = []
                b = []

            except NameError:
                print('Initialize all before altering parameters')

        else:
            print('Command not found. Type h for help.')