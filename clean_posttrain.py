import json
from pathlib import Path

propagator_root = Path('/home/user/propagator')

# 1. Clean propagator_posttrain_10k.jsonl
posttrain_path = propagator_root / 'data/propagator_posttrain_10k.jsonl'
cleaned_path = propagator_root / 'data/propagator_posttrain_cleaned.jsonl'

corrupt_tasks = {
    'arithmetic',
    'classification',
    'text_image_recognition',
    'image_recognition',
    'image_scene_reasoning'
}

print('Filtering propagator_posttrain_10k.jsonl...')
cleaned_rows = []
removed_counts = {}

with open(posttrain_path, 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        task = row.get('task', 'unknown')
        if task in corrupt_tasks:
            removed_counts[task] = removed_counts.get(task, 0) + 1
        else:
            cleaned_rows.append(row)

with open(cleaned_path, 'w', encoding='utf-8') as f:
    for row in cleaned_rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(f"Total remaining rows: {len(cleaned_rows)}")
print("Removed counts:")
for task, count in removed_counts.items():
    print(f"  - {task}: {count}")


# 2. Update propagator_posttrain_mix.json to point to propagator_posttrain_cleaned.jsonl
posttrain_mix_path = propagator_root / 'data/propagator_posttrain_mix.json'
with open(posttrain_mix_path, 'r', encoding='utf-8') as f:
    pt_mix = json.load(f)

for item in pt_mix:
    if item.get('data_files') == 'data/propagator_posttrain_10k.jsonl':
        item['data_files'] = 'data/propagator_posttrain_cleaned.jsonl'
        print("Updated propagator_posttrain_mix.json to point to cleaned file.")

with open(posttrain_mix_path, 'w', encoding='utf-8') as f:
    json.dump(pt_mix, f, ensure_ascii=False, indent=2)


# 3. Remove sample_05_format_following.jsonl from propagator_dataset_mix.json
# and redistribute its 0.05 weight to HuggingFaceM4/VQAv2
dataset_mix_path = propagator_root / 'data/propagator_dataset_mix.json'
with open(dataset_mix_path, 'r', encoding='utf-8') as f:
    mix = json.load(f)

new_mix = []
redistributed_weight = 0.0

for item in mix:
    if 'sample_05_format_following.jsonl' in item.get('data_files', ''):
        redistributed_weight = item.get('weight', 0.0)
        print(f"Removed sample_05_format_following.jsonl (weight {redistributed_weight}) from pretrain mix.")
    else:
        new_mix.append(item)

# Add the weight to VQAv2
for item in new_mix:
    if item.get('name') == 'HuggingFaceM4/VQAv2':
        old_w = item.get('weight', 0.0)
        item['weight'] = round(old_w + redistributed_weight, 4)
        print(f"Redistributed weight to HuggingFaceM4/VQAv2: {old_w} -> {item['weight']}")

with open(dataset_mix_path, 'w', encoding='utf-8') as f:
    json.dump(new_mix, f, ensure_ascii=False, indent=2)

print("Done cleaning datasets and configs!")
