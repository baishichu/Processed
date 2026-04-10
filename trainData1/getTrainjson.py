input_files = [
    'sammtrain.jsonl',
    'casme2train.jsonl',
    'smictrain_valid.jsonl',
]

output_file = 'train1.jsonl'

with open(output_file, 'w', encoding='utf-8') as fout:
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                fout.write(line)
print(f"saveAs {output_file}")
