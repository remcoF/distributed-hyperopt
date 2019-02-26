import json
import os


def main():
    test_dict = {'dog': 'bernese'}
    file_path = os.path.join('./logs', 'test_json.json')
    with open(file_path, 'w') as f:
        json.dump(json.dumps(test_dict), f)

    return


if __name__ == '__main__':
    main()