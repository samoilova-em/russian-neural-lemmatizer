import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.model import MorphModel
from src.dataset import ContextualMorphDataset
from src.data_utils import create_morph_dict, find_lemma_in_group, parse_annotated_corpus, build_vocabularies 
from src.train import train_model, save_checkpoint, evaluate


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Обучение нейросетевого морфологического анализатора русского языка"
    )
    
    # Пути к данным (с дефолтными значениями)
    parser.add_argument(
        "--xml-path",
        type=str,
        default="data/sample_annot.opcorpora.xml",
        help="Путь к XML-корпусу (по умолчанию: data/sample_annot.opcorpora.xml)"
    )
    parser.add_argument(
        "--dict-path",
        type=str,
        default="data/sample_dict.opcorpora.txt",
        help="Путь к морфологическому словарю (по умолчанию: data/sample_dict.opcorpora.txt)"
    )
    
    # Путь к чекпоинту
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/sample_morph_checkpoint.pth",
        help="Путь для сохранения чекпоинта (по умолчанию: checkpoints/sample_morph_checkpoint.pth)"
    )
    
    # Гиперпараметры
    parser.add_argument("--max-sentences", type=int, default=50000, help="Максимальное количество предложений")
    parser.add_argument("--max-word-len", type=int, default=30, help="Максимальная длина слова")
    parser.add_argument("--max-sent-len", type=int, default=20, help="Максимальная длина предложения")
    parser.add_argument("--batch-size", type=int, default=64, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=5, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    
    return parser.parse_args()


def main():
    # Получаем аргументы (если не переданы — используются значения по умолчанию)
    args = parse_args()
    
    # Проверяем существование файлов
    xml_path = Path(args.xml_path)
    dict_path = Path(args.dict_path)
    
    if not xml_path.exists():
        print(f"Ошибка: файл корпуса не найден: {xml_path}")
        print("Используйте --xml-path для указания правильного пути")
        return
    
    if not dict_path.exists():
        print(f"Ошибка: файл словаря не найден: {dict_path}")
        print("Используйте --dict-path для указания правильного пути")
        return
    
    # Создаём папку для чекпоинтов
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {DEVICE}")
    print(f"Корпус: {xml_path}")
    print(f"Словарь: {dict_path}")
    print(f"Чекпоинт будет сохранён: {checkpoint_path}")
    
    # Загрузка данных
    print("\nЗагрузка словаря...")
    morph_dict = create_morph_dict(str(dict_path))
    
    print("Построение словарей...")
    char2id, tag2id, id2tag = build_vocabularies()
    
    print(f"Парсинг корпуса (макс. {args.max_sentences} предложений)...")
    data = parse_annotated_corpus(
        str(xml_path), 
        max_sentences=args.max_sentences, 
        morph_dict=morph_dict
    )
    
    # Создание датасета
    dataset = ContextualMorphDataset(
        data, char2id, tag2id,
        max_word_len=args.max_word_len,
        max_sent_len=args.max_sent_len
    )
    
    # Разделение на выборки
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nРазмер датасета: {total_size} предложений")
    print(f"   Train: {train_size} ({train_size/total_size*100:.1f}%)")
    print(f"   Val:   {val_size} ({val_size/total_size*100:.1f}%)")
    print(f"   Test:  {test_size} ({test_size/total_size*100:.1f}%)")
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Инициализация модели
    model = MorphModel(
        char_vocab_size=len(char2id),
        char_emb_dim=128,
        hidden_dim=512,
        n_pos_tags=len(tag2id),
        max_word_len=args.max_word_len
    )
    
    print(f"\nПараметры модели: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Гиперпараметры: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    print("\nНачало обучения...")
    
    # Обучение
    model, history = train_model(
        model, train_loader, val_loader, test_loader, DEVICE,
        epochs=args.epochs, 
        lr=args.lr, 
        teacher_forcing_ratio=0.7
    )
    
    # Сохранение чекпоинта
    config = {
        'char_emb_dim': 128, 
        'hidden_dim': 512,
        'max_word_len': args.max_word_len, 
        'n_pos_tags': len(tag2id)
    }
    save_checkpoint(model, char2id, tag2id, id2tag, config, str(checkpoint_path))
    
    print("\nОбучение завершено!")


if __name__ == '__main__':
    main()