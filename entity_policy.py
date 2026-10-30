"""57-token relational policy for the pinned observe2 ABI.

This prototype uses card IDs and visible numeric features, not pretrained rule
text embeddings. It is a capacity experiment, not a reproduction of TavernAgent.
"""


def build_entity_policy(nn, torch, cfg):
    class EntityPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.transformer_dim
            if d % cfg.transformer_heads:
                raise ValueError("Transformer dimension must divide its head count")
            self.cards = nn.Embedding(cfg.card_vocab, d, padding_idx=0)
            self.global_projection = nn.Sequential(nn.LayerNorm(4017), nn.Linear(4017, d))
            self.item_projection = nn.Linear(55, d)
            self.opponent_projection = nn.Linear(75, d)
            self.positions = nn.Embedding(57, d)
            layer = nn.TransformerEncoderLayer(
                d, cfg.transformer_heads, dim_feedforward=4 * d,
                dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, cfg.transformer_layers,
                                                 norm=nn.LayerNorm(d))
            self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 245))
            self.query = nn.Linear(d, 7 * d)

        def forward(self, features, cards):
            f = features.float()
            f = f.sign() * torch.log1p(f.abs())
            cards = cards.clamp(0, cfg.card_vocab - 1)
            emb = self.cards(cards)
            b = f.shape[0]
            zeros = f.new_zeros
            board = f[:, 68:285].reshape(b, 7, 31)
            board_x = f[:, 3301:3364].reshape(b, 7, 9)
            board_n = f[:, 3364:3406].reshape(b, 7, 6)
            board = torch.cat([zeros((b, 7, 9)), board, board_x, board_n], -1)
            other_items = f[:, 285:1645].reshape(b, 34, 40)
            item_counts = f[:, 3406:3474].reshape(b, 34, 2)
            other_items = torch.cat([other_items, zeros((b, 34, 9)),
                                     item_counts, zeros((b, 34, 4))], -1)
            item_features = torch.cat([board, other_items], 1)
            # Count-weighted enchantment embeddings are attached to their owner.
            def attached(start, n, k, counts):
                ids = cards[:, start:start + n * k].reshape(b, n, k)
                values = emb[:, start:start + n * k].reshape(b, n, k, -1)
                weights = counts.unsqueeze(-1) * (ids != 0).unsqueeze(-1)
                return (values * weights).sum(2) / weights.sum(2).clamp_min(1)
            att = torch.cat([
                attached(124, 7, 6, board_n),
                attached(166, 34, 2, item_counts)], 1)
            items = self.item_projection(item_features) + emb[:, 19:60] + att
            opp = f[:, 1645:1708].reshape(b, 7, 9)
            seen = f[:, 1708:3227].reshape(b, 7, 7, 31)
            seen_mask = (cards[:, 75:124].reshape(b, 7, 7) != 0).unsqueeze(-1)
            seen_mean = (seen * seen_mask).sum(2) / seen_mask.sum(2).clamp_min(1)
            summaries = f[:, 3656:3901].reshape(b, 7, 35)
            opp = self.opponent_projection(torch.cat([opp, seen_mean, summaries], -1))
            opp = opp + emb[:, 68:75] + emb[:, 409:416]
            opp = opp + emb[:, 381:409].reshape(b, 7, 4, -1).sum(2) / 4
            # The global token retains every visible extension, including counters.
            global_token = self.global_projection(f).unsqueeze(1)
            global_token = global_token + emb.mean(1, keepdim=True)
            tokens = torch.cat([global_token, items, opp, emb[:, 60:68]], 1)
            present = torch.cat([
                torch.ones((b, 1), dtype=torch.bool, device=f.device),
                (cards[:, 19:60] != 0) | (item_features.abs().sum(-1) != 0),
                f[:, 1645:1708].reshape(b, 7, 9)[:, :, 0] != 0,
                cards[:, 60:68] != 0], 1)
            positions = torch.arange(57, device=f.device)
            encoded = self.encoder(tokens + self.positions(positions),
                                   src_key_padding_mask=~present)
            hidden = encoded[:, 0]
            contextual = encoded[:, 1:42]
            board, hand = contextual[:, :7], contextual[:, 7:17]
            shop, choices = contextual[:, 17:25], contextual[:, 25:41]
            q = self.query(hidden).reshape(b, 7, -1)
            def score(items, group):
                return (items * q[:, group:group + 1]).sum(-1) / cfg.transformer_dim ** 0.5
            prefix = zeros((b, 8))
            suffix = zeros((b, 49))
            play = score(hand, 4).unsqueeze(-1).expand(-1, -1, 8).reshape(b, 80)
            magnetic = score(hand, 5).unsqueeze(-1) + score(board, 6).unsqueeze(1)
            pointers = torch.cat([prefix, score(shop, 0), score(board, 1),
                                  score(board, 2), score(choices, 3), play,
                                  magnetic.reshape(b, 70), suffix], -1)
            return self.head(hidden) + pointers

    return EntityPolicy()
