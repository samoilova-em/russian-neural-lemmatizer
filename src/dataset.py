import torch
from torch.utils.data import Dataset

class ContextualMorphDataset(Dataset):
    def __init__(
        self,
        sentences,
        char2id,
        tag2id,
        max_word_len=30,
        max_sent_len=50
    ):
        self.sentences = sentences
        self.char2id = char2id
        self.tag2id = tag2id
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len

    def __len__(self):
        return len(self.sentences)

    def _encode_word(self, word, is_lemma=False):
        word = word.lower().replace('ё', 'е')

        limit = self.max_word_len - 1 if is_lemma else self.max_word_len
        ids = [self.char2id.get(char, 1) for char in word[:limit]]

        if is_lemma:
            ids.append(self.char2id['<EOS>'])

        ids += [0] * (self.max_word_len - len(ids))
        return ids

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        n_words = min(len(sentence), self.max_sent_len)

        input_chars = []
        target_pos = []
        target_lemma = []

        for i in range(self.max_sent_len):
            if i < n_words:
                word, lemma, pos = sentence[i]

                input_chars.append(self._encode_word(word, is_lemma=False))
                target_pos.append(self.tag2id.get(pos, 0))

                target_lemma.append(self._encode_word(lemma, is_lemma=True))
            else:
                input_chars.append([0] * self.max_word_len)
                target_pos.append(0)
                target_lemma.append([0] * self.max_word_len)

        return {
            'input_chars': torch.tensor(input_chars, dtype=torch.long),
            'target_pos': torch.tensor(target_pos, dtype=torch.long),
            'target_lemma': torch.tensor(target_lemma, dtype=torch.long),
            'length': n_words
        }