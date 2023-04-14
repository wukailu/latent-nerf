import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+')
    args = parser.parse_args()

    import jsonlines
    total = []
    for file in args.files:
        with jsonlines.open(file, 'r') as reader:
            total += [l for l in reader]

    with jsonlines.open("merged_jsonlines.jsonl", 'w') as writer:
        writer.write_all(total)