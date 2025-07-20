import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import threading
from pathlib import Path
import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
import time
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import pickle 
import json
import os
import math

class BERTEmbedding(nn.Module):
    _mappings_cache = None
    _cache_lock = threading.Lock()
    
    @classmethod
    def _load_mappings(cls):
        if cls._mappings_cache is None:
            with cls._cache_lock:
                if cls._mappings_cache is None:  # Double-checked locking
                    try:
                        main_dir = os.getcwd()
                        relative_path_dataset = "Data/preprocessed/AnimeRatings_min_rating7-min_uc10-min_sc10-splitleave_one_out/dataset.pkl"
                        relative_path_genres = "Data/AnimeRatings/id_to_genreids.json"

                        full_path_dataset = Path(main_dir) / relative_path_dataset
                        full_path_genres = Path(main_dir) / relative_path_genres

                        with full_path_dataset.open('rb') as f:
                            dataset_smap = pickle.load(f)["smap"]
                        
                        with full_path_genres.open('rb') as f:
                            id_to_genres = json.load(f)
                        
                        cls._mappings_cache = {
                            'dataset_smap': dataset_smap,
                            'id_to_genres': id_to_genres
                        }
                        
                    except Exception as e:
                        print(f"Warning: Could not load mappings: {e}")
                        cls._mappings_cache = {
                            'dataset_smap': {},
                            'id_to_genres': {}
                        }
        return cls._mappings_cache
    
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1, multi_genre=True, 
                 max_genres_per_anime=5, genre_weight=1.0):
        super().__init__()

        mappings = self._load_mappings()
        dataset_smap = mappings['dataset_smap']
        id_to_genres = mappings['id_to_genres']
        
        self.multi_genre = multi_genre
        self.max_genres_per_anime = max_genres_per_anime
        self.genre_weight = genre_weight  # Genre embedding'in etkisini kontrol etmek için

        all_genres = set()
        for anime_id, genres in id_to_genres.items():
            all_genres.update(genres)

        max_genre_id = max(all_genres) if all_genres else 0
        self.num_genres = max_genre_id + 1
        
        print(f"Detected {self.num_genres} unique genres (max_id: {max_genre_id})")
        print(f"Genre weight: {self.genre_weight}")

        self.vocab_size = vocab_size
        
        if multi_genre:
            self._create_multi_genre_mapping(dataset_smap, id_to_genres, vocab_size)
        else:
            self._create_single_genre_mapping(dataset_smap, id_to_genres, vocab_size)

        # Token embedding
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        
        # Genre embedding - özel başlatma ile
        self.genre_embed = nn.Embedding(num_embeddings=self.num_genres, embedding_dim=embed_size, padding_idx=0)
        
        # Xavier initialization for genre embeddings
        nn.init.xavier_uniform_(self.genre_embed.weight)
        if self.genre_embed.padding_idx is not None:
            nn.init.constant_(self.genre_embed.weight[self.genre_embed.padding_idx], 0)

        if multi_genre:
            # Learnable genre aggregation weights
            self.genre_aggregation = nn.Parameter(torch.ones(max_genres_per_anime) / max_genres_per_anime)
            
            # Genre attention mechanism
            self.genre_attention = nn.MultiheadAttention(embed_size, num_heads=4, batch_first=True)
            
            # Fusion layer with residual connection
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_size * 2, embed_size * 2),  # Genişletilmiş ara katman
                nn.LayerNorm(embed_size * 2),
                nn.GELU(),
                nn.Linear(embed_size * 2, embed_size),
                nn.LayerNorm(embed_size)
            )
            
            # Gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(embed_size * 2, embed_size),
                nn.Sigmoid()
            )
        else:
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_size * 2, embed_size * 2),
                nn.LayerNorm(embed_size * 2),
                nn.GELU(),
                nn.Linear(embed_size * 2, embed_size),
                nn.LayerNorm(embed_size)
            )
            
            # Gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(embed_size * 2, embed_size),
                nn.Sigmoid()
            )
        
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

        self._genre_cache = {}
        self._cache_lock = threading.Lock()
        
        # Debugging için
        self.register_buffer('debug_counter', torch.tensor(0))
    
    def _create_single_genre_mapping(self, dataset_smap, id_to_genres, vocab_size):
        token_to_genre = {}
        genre_coverage = 0
        
        for anime_id, token_id in dataset_smap.items():
            if token_id < vocab_size:
                genre_list = id_to_genres.get(str(anime_id), [0])
                genre_id = genre_list[0] if genre_list else 0
                if genre_id >= self.num_genres:
                    print(f"Warning: Genre ID {genre_id} >= {self.num_genres}, setting to 0")
                    genre_id = 0
                token_to_genre[token_id] = genre_id
                if genre_id != 0:
                    genre_coverage += 1
        
        print(f"Genre coverage: {genre_coverage}/{len(dataset_smap)} animes have genre info")
        
        if token_to_genre:
            token_ids = torch.tensor(list(token_to_genre.keys()), dtype=torch.long)
            genre_ids = torch.tensor(list(token_to_genre.values()), dtype=torch.long)
            
            self.register_buffer('token_ids', token_ids)
            self.register_buffer('genre_ids', genre_ids)
            self.has_mappings = True
        else:
            self.register_buffer('token_ids', torch.empty(0, dtype=torch.long))
            self.register_buffer('genre_ids', torch.empty(0, dtype=torch.long))
            self.has_mappings = False
    
    def _create_multi_genre_mapping(self, dataset_smap, id_to_genres, vocab_size):
        token_to_genres = {}
        genre_coverage = 0
        
        for anime_id, token_id in dataset_smap.items():
            if token_id < vocab_size:
                genre_list = id_to_genres.get(str(anime_id), [0])

                valid_genres = []
                for genre_id in genre_list:
                    if genre_id >= self.num_genres:
                        print(f"Warning: Genre ID {genre_id} >= {self.num_genres}, setting to 0")
                        genre_id = 0
                    valid_genres.append(genre_id)

                if len(valid_genres) < self.max_genres_per_anime:
                    valid_genres.extend([0] * (self.max_genres_per_anime - len(valid_genres)))
                else:
                    valid_genres = valid_genres[:self.max_genres_per_anime]
                
                token_to_genres[token_id] = valid_genres
                if any(g != 0 for g in valid_genres):
                    genre_coverage += 1
        
        print(f"Genre coverage: {genre_coverage}/{len(dataset_smap)} animes have genre info")
        
        if token_to_genres:
            token_ids = torch.tensor(list(token_to_genres.keys()), dtype=torch.long)
            genre_ids = torch.tensor(list(token_to_genres.values()), dtype=torch.long)
            
            self.register_buffer('token_ids', token_ids)
            self.register_buffer('genre_ids', genre_ids)
            self.has_mappings = True
        else:
            self.register_buffer('token_ids', torch.empty(0, dtype=torch.long))
            self.register_buffer('genre_ids', torch.empty(0, self.max_genres_per_anime, dtype=torch.long))
            self.has_mappings = False
    
    def _get_single_genre_mapping(self, sequence):
        """Original single genre mapping with improved bounds checking"""
        batch_size, seq_len = sequence.shape
        device = sequence.device
        
        if not self.has_mappings:
            return torch.zeros_like(sequence)

        sequence = torch.clamp(sequence, 0, self.vocab_size - 1)
        
        genre_sequence = torch.zeros_like(sequence)
        flat_sequence = sequence.flatten()
        flat_genre = torch.zeros_like(flat_sequence)
        
        token_mask = torch.isin(flat_sequence, self.token_ids)
        
        if token_mask.any():
            valid_tokens = flat_sequence[token_mask]

            # Cache'i devre dışı bırak - her seferinde fresh hesaplama
            sorted_indices = torch.argsort(self.token_ids)
            sorted_tokens = self.token_ids[sorted_indices]
            sorted_genres = self.genre_ids[sorted_indices]
            
            indices = torch.searchsorted(sorted_tokens, valid_tokens)
            indices = torch.clamp(indices, 0, len(sorted_tokens) - 1)
            exact_matches = sorted_tokens[indices] == valid_tokens
            
            genre_values = torch.where(
                exact_matches, 
                sorted_genres[indices], 
                torch.tensor(0, device=device, dtype=self.genre_ids.dtype)
            )
            
            flat_genre[token_mask] = genre_values
        
        return flat_genre.view(batch_size, seq_len)
    
    def _get_multi_genre_mapping(self, sequence):
        """Get multiple genres for each anime in sequence with bounds checking"""
        batch_size, seq_len = sequence.shape
        device = sequence.device
        
        if not self.has_mappings:
            return torch.zeros(batch_size, seq_len, self.max_genres_per_anime, device=device, dtype=torch.long)

        sequence = torch.clamp(sequence, 0, self.vocab_size - 1)

        genre_sequences = torch.zeros(batch_size, seq_len, self.max_genres_per_anime, device=device, dtype=torch.long)
        
        flat_sequence = sequence.flatten()
        flat_genres = torch.zeros(len(flat_sequence), self.max_genres_per_anime, device=device, dtype=torch.long)
        
        token_mask = torch.isin(flat_sequence, self.token_ids)
        
        if token_mask.any():
            valid_tokens = flat_sequence[token_mask]

            # Cache'i devre dışı bırak
            sorted_indices = torch.argsort(self.token_ids)
            sorted_tokens = self.token_ids[sorted_indices]
            sorted_genres = self.genre_ids[sorted_indices]
            
            indices = torch.searchsorted(sorted_tokens, valid_tokens)
            indices = torch.clamp(indices, 0, len(sorted_tokens) - 1)
            exact_matches = sorted_tokens[indices] == valid_tokens

            genre_values = sorted_genres[indices]

            valid_mask = token_mask.nonzero(as_tuple=True)[0]
            exact_valid_mask = valid_mask[exact_matches]
            
            flat_genres[exact_valid_mask] = genre_values[exact_matches]
        
        return flat_genres.view(batch_size, seq_len, self.max_genres_per_anime)
    
    def _aggregate_genre_embeddings(self, genre_embeddings):
        """Aggregate multiple genre embeddings per anime with attention"""
        batch_size, seq_len, max_genres, embed_size = genre_embeddings.shape
        
        # Reshape for attention
        genre_emb_flat = genre_embeddings.view(-1, max_genres, embed_size)  # (batch*seq, max_genres, embed_size)
        
        # Self-attention over genres
        attended_genres, _ = self.genre_attention(
            genre_emb_flat, genre_emb_flat, genre_emb_flat
        )  # (batch*seq, max_genres, embed_size)
        
        # Weighted aggregation - doğru boyutlar için einsum düzelt
        weights = F.softmax(self.genre_aggregation, dim=0)  # (max_genres,)
        weighted_genres = torch.einsum('bge,g->be', attended_genres, weights)  # (batch*seq, embed_size)
        
        return weighted_genres.view(batch_size, seq_len, embed_size)
    
    def forward(self, sequence):
        """Enhanced forward pass with debugging"""
        if sequence.max() >= self.vocab_size:
            print(f"Warning: Input contains tokens >= vocab_size ({self.vocab_size})")
            
        sequence = torch.clamp(sequence, 0, self.vocab_size - 1)

        # Token embedding
        token_emb = self.token(sequence)  # (batch, seq, embed_size)

        if self.multi_genre:
            genre_sequences = self._get_multi_genre_mapping(sequence)  # (batch, seq, max_genres)
            
            genre_sequences = torch.clamp(genre_sequences, 0, self.num_genres - 1)
            
            genre_embeddings = self.genre_embed(genre_sequences)  # (batch, seq, max_genres, embed_size)
            
            aggregated_genre_emb = self._aggregate_genre_embeddings(genre_embeddings)  # (batch, seq, embed_size)
            
            # Scale genre embedding
            aggregated_genre_emb = aggregated_genre_emb * self.genre_weight
            
            combined = torch.cat([token_emb, aggregated_genre_emb], dim=-1)
            
        else:
            genre_sequence = self._get_single_genre_mapping(sequence)
            
            genre_sequence = torch.clamp(genre_sequence, 0, self.num_genres - 1)
            
            genre_emb = self.genre_embed(genre_sequence)
            
            # Scale genre embedding
            genre_emb = genre_emb * self.genre_weight
            
            combined = torch.cat([token_emb, genre_emb], dim=-1)

        # Gated fusion
        gate_weights = self.gate(combined)  # (batch, seq, embed_size)
        fused = self.fusion_layer(combined)  # (batch, seq, embed_size)
        
        # Apply gating: interpolate between token_emb and fused
        output = gate_weights * fused + (1 - gate_weights) * token_emb
        
        return self.dropout(output)

    def clear_cache(self):
        """Clear internal caches to free GPU memory"""
        with self._cache_lock:
            self._genre_cache.clear()
    
    @classmethod
    def clear_global_cache(cls):
        """Clear global mappings cache"""
        with cls._cache_lock:
            cls._mappings_cache = None

    def get_genre_stats(self, sequence):
        """Debug fonksiyonu - genre mapping istatistiklerini döndürür"""
        if self.multi_genre:
            genre_sequences = self._get_multi_genre_mapping(sequence)
            non_zero_genres = (genre_sequences != 0).sum()
            total_possible = genre_sequences.numel()
            return f"Multi-genre: {non_zero_genres}/{total_possible} non-zero genres"
        else:
            genre_sequence = self._get_single_genre_mapping(sequence)
            non_zero_genres = (genre_sequence != 0).sum()
            total_tokens = genre_sequence.numel()
            return f"Single-genre: {non_zero_genres}/{total_tokens} non-zero genres"
