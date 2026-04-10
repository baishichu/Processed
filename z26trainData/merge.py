input_files = [
    'ROW/smictrain.jsonl',
    'ROW/sammtrain.jsonl',
    'ROW/casmetrain.jsonl'
]

output_file = 'ROW/aaROWTrain.jsonl'

with open(output_file, 'w', encoding='utf-8') as fout:
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                fout.write(line)
print(f"合并完成，保存到 {output_file}")
