"""
Интерфейс для инференса с использованием обученной модели.
Позволяет выполнять морфологический разбор предложений без необходимости
обучать модель с нуля.

Использование:
    python interface.py                          # интерактивный режим
    python interface.py --sentence "Текст"       # одно предложение
    python interface.py --checkpoint path.pth    # свой чекпоинт
"""
import argparse
import torch
from pathlib import Path

from src.predict import predict_sentence
from src.model import MorphModel
from src.data_utils import build_vocabularies


def parse_args():
    parser = argparse.ArgumentParser(
        description="Морфологический анализ предложений с помощью обученной модели"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/morph_checkpoint.pth",
        help="Путь к чекпоинту модели (по умолчанию: checkpoints/morph_checkpoint.pth)"
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default=None,
        help="Предложение для анализа. Если не указано — интерактивный режим."
    )
    return parser.parse_args()


def load_model(checkpoint_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Чекпоинт не найден: {checkpoint_path}\n"
            "Обучите модель через python main.py или укажите правильный путь через --checkpoint"
        )
    
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    
    model = MorphModel(
        char_vocab_size=len(checkpoint['char2id']),
        char_emb_dim=checkpoint['config']['char_emb_dim'],
        hidden_dim=checkpoint['config']['hidden_dim'],
        n_pos_tags=checkpoint['config']['n_pos_tags'],
        max_word_len=checkpoint['config']['max_word_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    char2id = checkpoint['char2id']
    tag2id = checkpoint['tag2id']
    id2tag = checkpoint['id2tag']
    
    return model, char2id, id2tag, device


def analyze_sentence(model, sentence, char2id, id2tag, device):
    result = predict_sentence(sentence, model, char2id, id2tag, device)
    return " ".join(result)


def interactive_mode(model, char2id, id2tag, device):
    print("Интерактивный режим. Введите предложение для анализа.")
    print("Для выхода введите 'exit' или 'quit'.\n")
    
    while True:
        try:
            sentence = input("Введите текст: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break
        
        if sentence.lower() in ('exit', 'quit', 'выход'):
            break
        
        if not sentence:
            continue
        
        result = analyze_sentence(model, sentence, char2id, id2tag, device)
        print(f"Результат: {result}\n")


def main():
    args = parse_args()
    
    print(f"Загрузка модели из: {args.checkpoint}")
    model, char2id, id2tag, device = load_model(args.checkpoint)
    print(f"Модель загружена. Устройство: {device}")
    
    if args.sentence:
        # Режим одного предложения
        result = analyze_sentence(model, args.sentence, char2id, id2tag, device)
        print(f"\nВход:  {args.sentence}")
        print(f"Выход: {result}")
    else:
        # Интерактивный режим
        print()
        interactive_mode(model, char2id, id2tag, device)


if __name__ == '__main__':
    main()