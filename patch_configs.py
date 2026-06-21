import json
from pathlib import Path

# Patch dataset mix
dataset_mix_path = Path('/home/user/propagator/data/propagator_dataset_mix.json')
with open(dataset_mix_path, 'r', encoding='utf-8') as f:
    mix = json.load(f)

# Find weight sum of the seed image recognition items to keep total weight consistent
replaced_weight = 0.0
new_mix = []
for item in mix:
    data_files = item.get('data_files', '')
    if 'propagator_image_recognition_seed' in data_files or 'propagator_text_image_seed' in data_files:
        replaced_weight += item.get('weight', 0.0)
    else:
        new_mix.append(item)

# Add VQAv2
vqav2_entry = {
    "name": "HuggingFaceM4/VQAv2",
    "split": "train",
    "validation_split": "validation",
    "mode": "image_recognition",
    "streaming": True,
    "weight": round(replaced_weight, 4)
}
new_mix.insert(3, vqav2_entry)  # Insert around the same position

with open(dataset_mix_path, 'w', encoding='utf-8') as f:
    json.dump(new_mix, f, ensure_ascii=False, indent=2)
print("Updated propagator_dataset_mix.json with weight:", round(replaced_weight, 4))


# Patch post-train mix
posttrain_mix_path = Path('/home/user/propagator/data/propagator_posttrain_mix.json')
with open(posttrain_mix_path, 'r', encoding='utf-8') as f:
    pt_mix = json.load(f)

replaced_pt_weight = 0.0
new_pt_mix = []
for item in pt_mix:
    data_files = item.get('data_files', '')
    if 'propagator_image_recognition_seed' in data_files or 'propagator_text_image_seed' in data_files:
        replaced_pt_weight += item.get('weight', 0.0)
    else:
        new_pt_mix.append(item)

vqav2_pt_entry = {
    "name": "HuggingFaceM4/VQAv2",
    "split": "train",
    "validation_split": "validation",
    "mode": "image_recognition",
    "streaming": True,
    "weight": round(replaced_pt_weight, 4)
}
new_pt_mix.insert(2, vqav2_pt_entry)

with open(posttrain_mix_path, 'w', encoding='utf-8') as f:
    json.dump(new_pt_mix, f, ensure_ascii=False, indent=2)
print("Updated propagator_posttrain_mix.json with weight:", round(replaced_pt_weight, 4))
