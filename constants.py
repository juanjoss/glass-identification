x_train_max = 0
y_train_max = 0

def get_x_train_max():
    return x_train_max

def set_x_train_max(x):
    if x > 0:
        global x_train_max
        x_train_max = x

def get_y_train_max():
    return y_train_max

def set_y_train_max(y):
    if y > 0:
        global y_train_max
        y_train_max = y