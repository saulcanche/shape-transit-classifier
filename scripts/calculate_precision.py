import sys

def main():
    try:
        with open('data/expected_list.txt', 'r') as f:
            expected = [l.strip() for l in f.readlines() if l.strip()]
    except FileNotFoundError:
        print("data/expected_list.txt not found")
        return

    try:
        with open('data/actual_list.txt', 'r') as f:
            actual = [l.strip() for l in f.readlines() if l.strip()]
    except FileNotFoundError:
        print("data/actual_list.txt not found")
        return

    correct = 0
    total = len(expected)
    
    print(f"Total Expected: {len(expected)}")
    print(f"Total Actual: {len(actual)}")
    
    for i in range(min(len(expected), len(actual))):
        if expected[i] == actual[i]:
            correct += 1
        else:
            print(f"Mismatch at {i}: Expected {expected[i]}, Actual {actual[i]}")
            
    if len(actual) > len(expected):
        for i in range(len(expected), len(actual)):
            print(f"Extra detection at {i}: {actual[i]}")
    elif len(expected) > len(actual):
        for i in range(len(actual), len(expected)):
            print(f"Missing detection at {i}: {expected[i]}")

    print(f"\nAccuracy: {correct}/{total} = {(correct/total)*100:.2f}%")

if __name__ == '__main__':
    main()
