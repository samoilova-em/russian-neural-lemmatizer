import re
import torch
from .model import MorphModel

def predict_sentence(sentence, model, char2id, id2tag, device, max_word_len=30, max_sent_len=20):
    model.eval()

    words = re.findall(r'[а-яА-ЯёЁ]+', sentence)
    n_words = len(words)

    if n_words == 0:
        return []

    char_ids = []
    for word in words:
        w_chars = [char2id.get(char, 1) for char in word.lower().replace('ё', 'е')[:max_word_len]]
        w_chars += [0] * (max_word_len - len(w_chars))
        char_ids.append(w_chars)

    for _ in range(max_sent_len - n_words):
        char_ids.append([0] * max_word_len)

    char_tensor = torch.tensor([char_ids], dtype=torch.long, device=device)

    BOS_ID = 35
    EOS_ID = 36

    with torch.no_grad():
        word_repr = model.encode_word(char_tensor)
        context_out, _ = model.context_lstm(word_repr)
        pos_logits = model.pos_head(context_out)
        pos_preds = torch.argmax(pos_logits[0][:n_words], dim=1).cpu().numpy()

        lemma_preds = []
        id2char = {idx: char for char, idx in char2id.items()}

        for i in range(n_words):
            word_lemma_chars = []
            word_context = context_out[0, i, :].unsqueeze(0).unsqueeze(0)

            decoder_hidden = (word_context, word_context)

            decoder_input = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)

            recent_chars = []
            original_len = len(words[i])

            for t in range(max_word_len):
                char_emb = model.char_embedding(decoder_input)
                decoder_input_with_context = torch.cat([char_emb, word_context], dim=2)

                decoder_out, decoder_hidden = model.lemma_decoder(
                    decoder_input_with_context, decoder_hidden
                )

                char_logits = model.lemma_fc(decoder_out.squeeze(1))
                char_pred = torch.argmax(char_logits, dim=1).item()

                if char_pred == EOS_ID:
                    break

                char = id2char.get(char_pred, '')
                recent_chars.append(char)
                if len(recent_chars) >= 4:
                    last_2 = ''.join(recent_chars[-2:])
                    if last_2 * 2 == ''.join(recent_chars[-4:]):
                        break
                if len(recent_chars) >= 3:
                    if recent_chars[-1] == recent_chars[-2] == recent_chars[-3]:
                        break
                if len(word_lemma_chars) >= original_len + 3:
                    break

                word_lemma_chars.append(char)
                decoder_input = torch.tensor([[char_pred]], dtype=torch.long, device=device)

            lemma_preds.append(''.join(word_lemma_chars))

    result = []
    for i, word in enumerate(words):
        pos = id2tag.get(pos_preds[i], 'S')
        lemma = lemma_preds[i] if lemma_preds[i] else words[i].lower()
        result.append(f"{word}{{{lemma}={pos}}}")

    return result