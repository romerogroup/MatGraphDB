import json

def main():
    with open('examples/sandbox/periodic_table.json', 'r') as f:
        data= json.load(f)

    print(len(data))

    print(data['H'])

if __name__ == '__main__':
    main()