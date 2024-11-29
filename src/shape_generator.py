from random import randint, choice, shuffle
from nn_constants import DATASET_DIR, MIN_SHAPE_VALUE, MAX_SHAPE_VALUE, SHAPE_COUNT, DATASET_NAME

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
    with open(join(DATASET_DIR, filename), "rb") as f:
        return pickle.load(f)

def shape_generator(n):
    l = []
    for _ in range(n):
        y = randint(0, 6)
        l.append(shape(y))
    return l
import pickle
from os.path import join
def many():
    
    for shc in [100, 500, 1000]:
        print(f"Creating dataset {shc} ...")
        SHAPE_COUNT = shc
        train_count = int(SHAPE_COUNT * 0.6)
        test_count = int(SHAPE_COUNT * 0.4)
        print(train_count, test_count)
        dataset_filename = f"treelines-{SHAPE_COUNT}-{MIN_SHAPE_VALUE}-{MAX_SHAPE_VALUE}.bin"
        dataset = {
            "train" : shape_generator(train_count),
            "test" : shape_generator(test_count)
        }
        with open(join(DATASET_DIR, dataset_filename), "wb") as f:
            pickle.dump(dataset, f)

def one():
    train_count = int(SHAPE_COUNT * 0.6)
    test_count = int(SHAPE_COUNT * 0.4)
    print(train_count, test_count)
    dataset = {
        "train" : shape_generator(train_count),
        "test" : shape_generator(test_count)
    }
    with open(join(DATASET_DIR, DATASET_NAME), "wb") as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    #many()
    one()
    print("done.")
