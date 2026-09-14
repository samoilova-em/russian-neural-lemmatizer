import torch
import torch.nn as nn
import torch.nn.functional as F

class MorphModel(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim=128, hidden_dim=512,
                 n_pos_tags=10, max_word_len=30):
        super().__init__()

        self.char_vocab_size = char_vocab_size
        self.char_emb_dim = char_emb_dim
        self.hidden_dim = hidden_dim
        self.n_pos_tags = n_pos_tags
        self.max_word_len = max_word_len
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 35
        self.eos_id = 36

        self.char_embedding = nn.Embedding(
            char_vocab_size, char_emb_dim, padding_idx=0
        )

        self.char_encoder = nn.LSTM(
            char_emb_dim, hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=1
        )

        self.context_lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=1
        )

        self.pos_head = nn.Linear(hidden_dim, n_pos_tags)

        self.lemma_decoder = nn.LSTM(
            hidden_dim + char_emb_dim, hidden_dim,
            batch_first=True,
            num_layers=1
        )
        self.lemma_fc = nn.Linear(hidden_dim, char_vocab_size)

        self.dropout = nn.Dropout(0.3)

    def encode_word(self, char_ids):
        batch_size, seq_len, word_len = char_ids.shape

        char_emb = self.char_embedding(char_ids)
        char_emb = char_emb.view(batch_size * seq_len, word_len, -1)

        _, (char_hidden, _) = self.char_encoder(char_emb)
        char_hidden = char_hidden.transpose(0, 1).contiguous()
        char_hidden = char_hidden.view(batch_size, seq_len, -1)

        return char_hidden

    def forward(self, char_ids, lemma_ids=None, teacher_forcing_ratio=0.5):
      batch_size, seq_len, _ = char_ids.shape

      word_repr = self.encode_word(char_ids)
      word_repr = self.dropout(word_repr)

      context_out, _ = self.context_lstm(word_repr)
      context_out = self.dropout(context_out)

      pos_logits = self.pos_head(context_out)

      lemma_logits = torch.zeros(
          batch_size, seq_len, self.max_word_len, self.char_vocab_size,
          device=char_ids.device
      )

      for i in range(seq_len):
          word_context = context_out[:, i, :]

          decoder_hidden = (
              word_context.unsqueeze(0),
              word_context.unsqueeze(0)
          )

          decoder_input = torch.full(
              (batch_size, 1), self.bos_id,
              dtype=torch.long, device=char_ids.device
          )

          for t in range(self.max_word_len):
              char_emb = self.char_embedding(decoder_input)

              decoder_input_with_context = torch.cat(
                  [char_emb, word_context.unsqueeze(1)], dim=2
              )

              decoder_out, decoder_hidden = self.lemma_decoder(
                  decoder_input_with_context, decoder_hidden
              )

              char_logits = self.lemma_fc(decoder_out.squeeze(1))
              lemma_logits[:, i, t, :] = char_logits

              if lemma_ids is not None and torch.rand(1).item() < teacher_forcing_ratio:
                  next_char = lemma_ids[:, i, t:t+1]
              else:
                  next_char = torch.argmax(char_logits, dim=1, keepdim=True)

              decoder_input = next_char

      return pos_logits, lemma_logits