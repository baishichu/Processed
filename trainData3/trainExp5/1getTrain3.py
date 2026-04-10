input_files = [
    '1sammtrain3.jsonl',
    '1casme2train3.jsonl'
]

output_file = '1pro_train3.jsonl'

with open(output_file, 'w', encoding='utf-8') as fout:
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                fout.write(line)
print(f"saveAs {output_file}")
