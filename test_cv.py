import datasets
import time
import numpy as np
import sys
sys.path.append('/home/user/propagator')
import train

print('Loading common_voice_17_0...')
try:
    ds = datasets.load_dataset('fixie-ai/common_voice_17_0', 'en', split='train', streaming=True)
    it = iter(ds)
    
    # Initialize train globals
    train.config = train.build_config()
    train.tokenizer = train.Tokenizer.from_file(train.config.tokenizer_path)
    train.token_ids = train.ensure_special_tokens(train.tokenizer)
    train.text_vocab_size = train.tokenizer.get_vocab_size()
    _, train.audio_token_start, train.audio_token_end, train.image_token_start, train.image_token_end = train.compute_vocab_sizes(train.text_vocab_size)
    train.init_global_token_ids()

    rows = []
    print('Fetching 5 rows...')
    start_fetch = time.time()
    for i in range(5):
        row = next(it)
        rows.append(row)
    end_fetch = time.time()
    print(f"Time to fetch 5 rows: {end_fetch - start_fetch:.2f} seconds")

    print('Encoding 5 rows with Mimi...')
    start_encode = time.time()
    for i, row in enumerate(rows):
        audio = train.extract_audio_array(row, None)
        tokens = train.encode_audio_batch_to_token_ids([audio])[0]
        print(f"Row {i} encoded into {len(tokens)} token frames.")
    end_encode = time.time()
    print(f"Time to encode 5 rows: {end_encode - start_encode:.2f} seconds")

except Exception as e:
    import traceback
    traceback.print_exc()
