import os
import torch
import torch.nn as nn
from .model import MorphModel

def evaluate(model, loader, device, criterion_pos, criterion_lemma):
    model.eval()
    total_loss_pos, total_loss_lemma = 0.0, 0.0
    correct_pos, total_pos = 0, 0
    correct_lemma, total_lemma = 0, 0

    with torch.no_grad():
        for batch in loader:
            char_ids = batch['input_chars'].to(device)
            pos_ids = batch['target_pos'].to(device)
            lemma_ids = batch['target_lemma'].to(device)

            pos_logits, lemma_logits = model(char_ids, lemma_ids, teacher_forcing_ratio=0.0)

            b, s, n_tags = pos_logits.shape
            loss_pos = criterion_pos(pos_logits.view(-1, n_tags), pos_ids.view(-1))

            b, s, max_len, vocab = lemma_logits.shape
            loss_lemma = criterion_lemma(lemma_logits.view(-1, vocab), lemma_ids.view(-1))

            total_loss_pos += loss_pos.item()
            total_loss_lemma += loss_lemma.item()

            _, predicted = torch.max(pos_logits, 2)
            mask = pos_ids != 0
            correct_pos += (predicted == pos_ids)[mask].sum().item()
            total_pos += mask.sum().item()

            _, lemma_predicted = torch.max(lemma_logits, 3)
            lemma_mask = lemma_ids != 0
            correct_lemma += (lemma_predicted == lemma_ids)[lemma_mask].sum().item()
            total_lemma += lemma_mask.sum().item()

    acc_pos = correct_pos / total_pos * 100 if total_pos > 0 else 0
    acc_lemma = correct_lemma / total_lemma * 100 if total_lemma > 0 else 0
    loss_pos = total_loss_pos / len(loader)
    loss_lemma = total_loss_lemma / len(loader)

    return loss_pos, loss_lemma, acc_pos, acc_lemma
    
def train_model(model, train_loader, val_loader, test_loader, device, epochs=10, lr=0.0005, teacher_forcing_ratio=0.7):
    criterion_pos = nn.CrossEntropyLoss(ignore_index=0)
    vocab_size = model.char_vocab_size
    token_weights = torch.ones(vocab_size, device=device)
    criterion_lemma = nn.CrossEntropyLoss(ignore_index=0, weight=token_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    model.to(device)

    history = {
        'train_loss_pos': [], 'train_loss_lemma': [],
        'train_acc_pos': [], 'train_acc_lemma': [],
        'val_loss_pos': [], 'val_loss_lemma': [],
        'val_acc_pos': [], 'val_acc_lemma': []
    }

    for epoch in range(epochs):
        model.train()
        total_loss_pos, total_loss_lemma = 0.0, 0.0
        correct_pos, total_pos = 0, 0
        correct_lemma, total_lemma = 0, 0

        for batch in train_loader:
            char_ids = batch['input_chars'].to(device)
            pos_ids = batch['target_pos'].to(device)
            lemma_ids = batch['target_lemma'].to(device)

            optimizer.zero_grad()
            pos_logits, lemma_logits = model(char_ids, lemma_ids, teacher_forcing_ratio=teacher_forcing_ratio)

            b, s, n_tags = pos_logits.shape
            loss_pos = criterion_pos(pos_logits.view(-1, n_tags), pos_ids.view(-1))
            b, s, max_len, vocab = lemma_logits.shape
            loss_lemma = criterion_lemma(lemma_logits.view(-1, vocab), lemma_ids.view(-1))

            loss = loss_pos + loss_lemma
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_pos += loss_pos.item()
            total_loss_lemma += loss_lemma.item()

            _, predicted = torch.max(pos_logits, 2)
            mask = pos_ids != 0
            correct_pos += (predicted == pos_ids)[mask].sum().item()
            total_pos += mask.sum().item()

            _, lemma_predicted = torch.max(lemma_logits, 3)
            lemma_mask = lemma_ids != 0
            correct_lemma += (lemma_predicted == lemma_ids)[lemma_mask].sum().item()
            total_lemma += lemma_mask.sum().item()

        train_loss_pos = total_loss_pos / len(train_loader)
        train_loss_lemma = total_loss_lemma / len(train_loader)
        train_acc_pos = correct_pos / total_pos * 100 if total_pos > 0 else 0
        train_acc_lemma = correct_lemma / total_lemma * 100 if total_lemma > 0 else 0

        val_loss_pos, val_loss_lemma, val_acc_pos, val_acc_lemma = evaluate(
            model, val_loader, device, criterion_pos, criterion_lemma
        )

        scheduler.step(val_loss_lemma)

        history['train_loss_pos'].append(train_loss_pos)
        history['train_loss_lemma'].append(train_loss_lemma)
        history['train_acc_pos'].append(train_acc_pos)
        history['train_acc_lemma'].append(train_acc_lemma)
        history['val_loss_pos'].append(val_loss_pos)
        history['val_loss_lemma'].append(val_loss_lemma)
        history['val_acc_pos'].append(val_acc_pos)
        history['val_acc_lemma'].append(val_acc_lemma)

        print(f"Epoch {epoch + 1}/{epochs} | Train POS: {train_acc_pos:.2f}% | Val POS: {val_acc_pos:.2f}% | "
              f"Train Lemma: {train_acc_lemma:.2f}% | Val Lemma: {val_acc_lemma:.2f}%")

    print("\nИтоговая оценка")
    test_loss_pos, test_loss_lemma, test_acc_pos, test_acc_lemma = evaluate(
        model, test_loader, device, criterion_pos, criterion_lemma
    )
    print(f"Test POS Accuracy: {test_acc_pos:.2f}%")
    print(f"Test Lemma Accuracy: {test_acc_lemma:.2f}%")

    return model, history

def save_checkpoint(model, char2id, tag2id, id2tag, config, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'char2id': char2id,
        'tag2id': tag2id,
        'id2tag': id2tag,
        'config': config
    }
    torch.save(checkpoint, path)
    print(f"Чекпоинт сохранен {path}")


def load_checkpoint(path, device='cpu'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Чекпоинт не найден: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = MorphModel(
        char_vocab_size=len(checkpoint['char2id']),
        char_emb_dim=config['char_emb_dim'],
        hidden_dim=config['hidden_dim'],
        n_pos_tags=len(checkpoint['tag2id']),
        max_word_len=config['max_word_len']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return (
        model,
        checkpoint['char2id'],
        checkpoint['tag2id'],
        checkpoint['id2tag'],
        config
    )