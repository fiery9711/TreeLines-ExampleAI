from random import randint, choice, shuffle
from constants import DATASET_DIR, MIN_SHAPE_VALUE, MAX_SHAPE_VALUE, SHAPE_COUNT, DATASET_NAME

def shape(yk):
    rand = lambda: randint(MIN_SHAPE_VALUE, MAX_SHAPE_VALUE)
    a = sorted([rand(), rand(), rand()])
    match yk:
        case 0:
            x = [0, 0, 0]
        case 1:
            x1 = choice(a)
            x = [x1, 0, 0]
        case 2:
            x1 = choice(a)
            x = [x1, 0, x1]
        case 3:
            x1, _, x2 = a
            if x1 == x2:
                x2 += 1
            x = [x1, 0, x2]
        case 4:
            x1 = choice(a)
            x = [x1, x1, x1]
        case 5:
            x1, _, x2 = a
            if x1 == x2:
                x2 += 1
            x = [x1, x2, x1]
        case 6:
            x1, x2, x3 = a
            if x1 == x2:
                x2 += 1
            if x2 == x3:
                x3 += 1
            x = [x1, x2, x3]
        case _:
            x = a
    shuffle(x)
    return x, yk

def load(filename = DATASET_NAME):
    import pickle
    from os.path import join
    with open(join(DATASET_DIR, DATASET_NAME), "rb") as f:
        return pickle.load(f)

def shape_generator(n):
    l = []
    for _ in range(n):
        y = randint(0, 6)
        l.append(shape(y))
    return l

if __name__ == "__main__":
    import pickle
    from os.path import join
    print("Creating dataset ...")
    train_count = SHAPE_COUNT // 100 * 70
    val_count = SHAPE_COUNT // 100 * 25
    test_count = SHAPE_COUNT // 100 * 30
    dataset = {
        "train" : shape_generator(train_count),
        "test" : shape_generator(test_count)
    }
    with open(join(DATASET_DIR, DATASET_NAME), "wb") as f:
        pickle.dump(dataset, f)
    print("done.")
