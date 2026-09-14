from collections import Counter
from torch.utils.data import Dataset, DataLoader
from lxml import etree
import torch
import torch.nn as nn
import re

def create_morph_dict(filename):
    morph_dict = {}
    groups = []
    current_group = []
    group_id = None

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.isdigit():
                if current_group:
                    groups.append((group_id, current_group))
                group_id = int(line)
                current_group = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    current_group.append((parts[0], parts[1]))

        if current_group:
            groups.append((group_id, current_group))

    print(f"Найдено групп: {len(groups)}")

    for i, (gid, group) in enumerate(groups):
        has_infinitive = any(tag.startswith('INFN') for _, tag in group)

        if has_infinitive:
            infinitive = None
            for form, tag in group:
                if tag.startswith('INFN'):
                    infinitive = form
                    break

            if infinitive:
                for form, tag in group:
                    pos = tag.split(',')[0].split()[0]
                    key = form.upper()
                    if key not in morph_dict:
                        morph_dict[key] = (infinitive.upper(), pos)

        else:
            if i + 1 < len(groups):
                next_gid, next_group = groups[i + 1]
                next_has_infinitive = any(tag.startswith('INFN') for _, tag in next_group)

                if next_has_infinitive:
                    infinitive = None
                    for form, tag in next_group:
                        if tag.startswith('INFN'):
                            infinitive = form
                            break

                    if infinitive:
                        for form, tag in group:
                            pos = tag.split(',')[0].split()[0]
                            key = form.upper()
                            if key not in morph_dict:
                                morph_dict[key] = (infinitive.upper(), pos)
                        continue

            lemma = find_lemma_in_group(group)
            if lemma:
                for form, tag in group:
                    pos = tag.split(',')[0].split()[0]
                    key = form.upper()
                    if key not in morph_dict:
                        morph_dict[key] = (lemma.upper(), pos)

    print(f"Загружено словоформ: {len(morph_dict)}")
    return morph_dict

def find_lemma_in_group(group):
    for form, tag in group:
        if 'CONJ' in tag or 'PREP' in tag:
            return form

    for form, tag in group:
        if 'NOUN' in tag and 'nomn' in tag and 'sing' in tag:
            return form

    for form, tag in group:
        if 'ADJF' in tag and 'nomn' in tag and 'masc' in tag and 'sing' in tag:
            return form

    for form, tag in group:
        if 'NPRO' in tag and 'nomn' in tag:
            return form

    if group:
        return group[0][0]

    return None

def parse_annotated_corpus(xml_path, max_sentences=None, morph_dict=None):
    """
    morph_dict: словарь из create_morph_dict(),
                формат: {WORD_FORM_UPPER: (LEMMA_UPPER, POS)}
    """
    tag_map = {
        'NOUN': 'S', 'VERB': 'V', 'ADJF': 'A', 'ADJS': 'A',
        'PTCP': 'A', 'PRTF': 'A', 'PRTS': 'A', 'COMP': 'A',
        'ADV': 'ADV', 'ADVB': 'ADV', 'GRND': 'ADV',
        'PREP': 'PR', 'CONJ': 'CONJ',
        'PRON': 'NI', 'NPRO': 'NI', 'NUMR': 'NI',
        'PRED': 'PRED', 'INTJ': 'INTJ', 'INFN': 'V',
        'PNCT': 'PNCT'
    }

    sentences = []
    current_sentence = []
    sentence_count = 0

    context = etree.iterparse(xml_path, events=('end',), tag='token', recover=True)

    try:
        for event, elem in context:
            word = elem.get('text')
            if not word:
                elem.clear()
                continue

            lemma_elem = elem.find('.//l')
            if lemma_elem is not None:
                lemma = lemma_elem.get('t')

                pos_raw = None
                for gram in lemma_elem.findall('g'):
                    val = gram.get('v')
                    if val and val.isupper() and 3 <= len(val) <= 6:
                        pos_raw = val
                        break

                if lemma and pos_raw:
                    pos_tag = tag_map.get(pos_raw, 'S')

                    if pos_tag == 'V' and morph_dict:
                        word_key = word.upper()
                        if word_key in morph_dict:
                            final_lemma, _ = morph_dict[word_key]
                            final_lemma = final_lemma.lower()
                        else:
                            final_lemma = lemma
                    else:
                        final_lemma = lemma

                    current_sentence.append((word, final_lemma, pos_tag))

            if word in ['.', '!', '?', '…', '...']:
                if len(current_sentence) >= 3:
                    sentences.append(current_sentence)
                    sentence_count += 1
                current_sentence = []

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

            if max_sentences and sentence_count >= max_sentences:
                break

    except Exception as e:
        print(f"Ошибка парсинга: {e}")

    if current_sentence and len(current_sentence) >= 3:
        sentences.append(current_sentence)

    print(f"Извлечено предложений: {len(sentences):,}")
    return sentences

def build_vocabularies():
    char_vocab = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

    char2id = {char: idx + 2 for idx, char in enumerate(char_vocab)}

    char2id['<PAD>'] = 0
    char2id['<UNK>'] = 1
    char2id['<BOS>'] = 35
    char2id['<EOS>'] = 36

    all_tags = ['<PAD>', '<UNK>', 'S', 'V', 'A', 'ADV', 'PR', 'CONJ', 'NI', 'PRED', 'INTJ', 'PNCT']

    tag2id = {tag: idx for idx, tag in enumerate(all_tags)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}

    return char2id, tag2id, id2tag