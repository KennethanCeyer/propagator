import sys
from pathlib import Path

train_path = Path('/home/user/propagator/train.py')
content = train_path.read_text(encoding='utf-8')

# Patch _extract_image_value
target_1 = 'keys.extend(["image", "frame", "camera_image", "camera_frame", "pixels"])'
replacement_1 = 'keys.extend(["image", "images", "frame", "camera_image", "camera_frame", "pixels"])'

# Patch _image_value_to_array
target_2 = '''def _image_value_to_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):'''
replacement_2 = '''def _image_value_to_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    if isinstance(value, np.ndarray):'''

# Patch tokenize_image_recognition
target_3 = '''    image_text_keys = [image_key] if image_key else []
    image_text_keys.extend(["image_text", "caption", "description", "scene", "objects", "text"])
    question_keys = [question_key] if question_key else []
    question_keys.extend(["question", "prompt", "query"])
    answer_keys = [answer_key] if answer_key else []
    answer_keys.extend(["answer", "response", "label", "caption"])

    image_text = first_nonempty_string(row, image_text_keys)
    question = first_nonempty_string(row, question_keys) or "Describe the image."
    answer = first_nonempty_string(row, answer_keys)
    if not image_text:
        raise DataQualityError("Image recognition row has no image description or metadata text")'''

replacement_3 = '''    image_text_keys = [image_key] if image_key else []
    image_text_keys.extend(["image_text", "caption", "description", "scene", "objects", "text"])
    question_keys = [question_key] if question_key else []
    question_keys.extend(["question", "prompt", "query"])
    answer_keys = [answer_key] if answer_key else []
    answer_keys.extend(["answer", "response", "label", "caption", "multiple_choice_answer"])

    image_text = first_nonempty_string(row, image_text_keys)
    question = first_nonempty_string(row, question_keys) or "Describe the image."
    answer = first_nonempty_string(row, answer_keys)
    if not image_text:
        if _extract_image_value(row, spec) is not None:
            image_text = "image"
    if not image_text:
        raise DataQualityError("Image recognition row has no image description or metadata text")'''

content_patched = content
for t, r in [(target_1, replacement_1), (target_2, replacement_2), (target_3, replacement_3)]:
    if t not in content_patched:
        print(f"ERROR: Target not found:\n{t}")
        sys.exit(1)
    content_patched = content_patched.replace(t, r)

train_path.write_text(content_patched, encoding='utf-8')
print("Successfully patched train.py!")
