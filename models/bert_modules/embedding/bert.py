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


class BERTEmbedding(nn.Module):
    """
    GPU-optimized version with per-anime genre processing capability
    Fixed version with proper bounds checking and error handling
    """
    
    # Class-level cache for shared data
    _mappings_cache = None
    _cache_lock = threading.Lock()
    
    @classmethod
    def _load_mappings(cls):
        """Thread-safe singleton pattern for loading mappings once"""
        if cls._mappings_cache is None:
            with cls._cache_lock:
                if cls._mappings_cache is None:  # Double-checked locking
                    try:
                        with Path("/kaggle/working/AnimeRecBERT/Data/preprocessed/AnimeRatings54M_min_rating7-min_uc10-min_sc10-splitleave_one_out/dataset.pkl").open('rb') as f:
                            dataset_smap = pickle.load(f)["smap"]
                        
                        with open('/kaggle/working/AnimeRecBERT/Data/AnimeRatings54M/id_to_genreids.json', 'r', encoding='utf-8') as f:
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
    
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1, multi_genre=True, max_genres_per_anime=5):
        super().__init__()
        
        # Load shared mappings
        mappings = self._load_mappings()
        dataset_smap = mappings['dataset_smap']
        id_to_genres = mappings['id_to_genres']
        
        self.multi_genre = multi_genre
        self.max_genres_per_anime = max_genres_per_anime
        
        # Calculate actual genre count to prevent index errors
        all_genres = set()
        for anime_id, genres in id_to_genres.items():
            all_genres.update(genres)
        
        # Add padding genre (0) and calculate max genre id
        max_genre_id = max(all_genres) if all_genres else 0
        self.num_genres = max_genre_id + 1
        
        print(f"Detected {self.num_genres} unique genres (max_id: {max_genre_id})")
        
        # Store vocab_size for bounds checking
        self.vocab_size = vocab_size
        
        # Create mapping for tokens to genres
        if multi_genre:
            # Multi-genre processing: store all genres for each anime
            self._create_multi_genre_mapping(dataset_smap, id_to_genres, vocab_size)
        else:
            # Single genre processing (original behavior)
            self._create_single_genre_mapping(dataset_smap, id_to_genres, vocab_size)
        
        # Embeddings with proper size
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.genre_embed = nn.Embedding(num_embeddings=self.num_genres, embedding_dim=embed_size, padding_idx=0)
        
        # Fusion layer configuration based on mode
        if multi_genre:
            # For multi-genre: token + aggregated genre embeddings
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_size * 2, embed_size),
                nn.LayerNorm(embed_size),
                nn.ReLU()
            )
            
            # Genre aggregation methods
            self.genre_aggregation = nn.Parameter(torch.ones(max_genres_per_anime) / max_genres_per_anime)
            self.genre_attention = nn.MultiheadAttention(embed_size, num_heads=4, batch_first=True)
        else:
            # Original single genre fusion
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_size * 2, embed_size),
                nn.LayerNorm(embed_size),
                nn.ReLU()
            )
        
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        
        # Pre-allocate tensors for reuse (thread-safe)
        self._genre_cache = {}
        self._cache_lock = threading.Lock()
    
    def _create_single_genre_mapping(self, dataset_smap, id_to_genres, vocab_size):
        """Create mapping for single genre per anime (original behavior)"""
        token_to_genre = {}
        for token_id, anime_id in dataset_smap.items():
            if token_id < vocab_size:
                genre_list = id_to_genres.get(str(anime_id), [0])
                # Ensure genre ID is within bounds
                genre_id = genre_list[0] if genre_list else 0
                if genre_id >= self.num_genres:
                    print(f"Warning: Genre ID {genre_id} >= {self.num_genres}, setting to 0")
                    genre_id = 0
                token_to_genre[token_id] = genre_id
        
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
        """Create mapping for multiple genres per anime"""
        token_to_genres = {}
        for token_id, anime_id in dataset_smap.items():
            if token_id < vocab_size:
                genre_list = id_to_genres.get(str(anime_id), [0])
                
                # Validate and clamp genre IDs
                valid_genres = []
                for genre_id in genre_list:
                    if genre_id >= self.num_genres:
                        print(f"Warning: Genre ID {genre_id} >= {self.num_genres}, setting to 0")
                        genre_id = 0
                    valid_genres.append(genre_id)
                
                # Pad or truncate to max_genres_per_anime
                if len(valid_genres) < self.max_genres_per_anime:
                    valid_genres.extend([0] * (self.max_genres_per_anime - len(valid_genres)))
                else:
                    valid_genres = valid_genres[:self.max_genres_per_anime]
                
                token_to_genres[token_id] = valid_genres
        
        if token_to_genres:
            token_ids = torch.tensor(list(token_to_genres.keys()), dtype=torch.long)
            # Shape: (num_tokens, max_genres_per_anime)
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
        
        # Clamp input sequence to valid range
        sequence = torch.clamp(sequence, 0, self.vocab_size - 1)
        
        genre_sequence = torch.zeros_like(sequence)
        flat_sequence = sequence.flatten()
        flat_genre = torch.zeros_like(flat_sequence)
        
        # Use safer approach with explicit bounds checking
        token_mask = torch.isin(flat_sequence, self.token_ids)
        
        if token_mask.any():
            valid_tokens = flat_sequence[token_mask]
            
            # Thread-safe cache access
            with self._cache_lock:
                cache_key = (device, len(self.token_ids))
                if cache_key not in self._genre_cache:
                    sorted_indices = torch.argsort(self.token_ids)
                    self._genre_cache[cache_key] = {
                        'sorted_tokens': self.token_ids[sorted_indices],
                        'sorted_genres': self.genre_ids[sorted_indices]
                    }
                
                cached_data = self._genre_cache[cache_key]
            
            indices = torch.searchsorted(cached_data['sorted_tokens'], valid_tokens)
            indices = torch.clamp(indices, 0, len(cached_data['sorted_tokens']) - 1)
            exact_matches = cached_data['sorted_tokens'][indices] == valid_tokens
            
            genre_values = torch.where(
                exact_matches, 
                cached_data['sorted_genres'][indices], 
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
        
        # Clamp input sequence to valid range
        sequence = torch.clamp(sequence, 0, self.vocab_size - 1)
        
        # Output shape: (batch_size, seq_len, max_genres_per_anime)
        genre_sequences = torch.zeros(batch_size, seq_len, self.max_genres_per_anime, device=device, dtype=torch.long)
        
        flat_sequence = sequence.flatten()
        flat_genres = torch.zeros(len(flat_sequence), self.max_genres_per_anime, device=device, dtype=torch.long)
        
        token_mask = torch.isin(flat_sequence, self.token_ids)
        
        if token_mask.any():
            valid_tokens = flat_sequence[token_mask]
            
            # Thread-safe cache access
            with self._cache_lock:
                cache_key = (device, len(self.token_ids), 'multi')
                if cache_key not in self._genre_cache:
                    sorted_indices = torch.argsort(self.token_ids)
                    self._genre_cache[cache_key] = {
                        'sorted_tokens': self.token_ids[sorted_indices],
                        'sorted_genres': self.genre_ids[sorted_indices]  # Shape: (num_tokens, max_genres_per_anime)
                    }
                
                cached_data = self._genre_cache[cache_key]
            
            indices = torch.searchsorted(cached_data['sorted_tokens'], valid_tokens)
            indices = torch.clamp(indices, 0, len(cached_data['sorted_tokens']) - 1)
            exact_matches = cached_data['sorted_tokens'][indices] == valid_tokens
            
            # Get genre values for exact matches
            genre_values = cached_data['sorted_genres'][indices]  # Shape: (num_valid_tokens, max_genres_per_anime)
            
            # Only set genres for exact matches
            valid_mask = token_mask.nonzero(as_tuple=True)[0]
            exact_valid_mask = valid_mask[exact_matches]
            
            flat_genres[exact_valid_mask] = genre_values[exact_matches]
        
        return flat_genres.view(batch_size, seq_len, self.max_genres_per_anime)
    
    def _aggregate_genre_embeddings(self, genre_embeddings):
        """Aggregate multiple genre embeddings per anime"""
        # genre_embeddings shape: (batch_size, seq_len, max_genres_per_anime, embed_size)
        batch_size, seq_len, max_genres, embed_size = genre_embeddings.shape
        
        # Method 1: Weighted average
        weights = F.softmax(self.genre_aggregation, dim=0)
        weighted_genres = torch.einsum('bsgd,g->bsd', genre_embeddings, weights)
        
        return weighted_genres
    
    def forward(self, sequence):
        """
        Enhanced forward pass with per-anime genre processing
        """
        # Input validation and bounds checking
        if sequence.max() >= self.vocab_size:
            print(f"Warning: Input contains tokens >= vocab_size ({self.vocab_size})")
            
        sequence = torch.clamp(sequence, 0, self.vocab_size - 1)
        
        # 1. Token embeddings
        token_emb = self.token(sequence)
        
        # 2. Genre processing based on mode
        if self.multi_genre:
            # Multi-genre processing
            genre_sequences = self._get_multi_genre_mapping(sequence)  # (batch, seq, max_genres)
            
            # Ensure genre sequences are within bounds
            genre_sequences = torch.clamp(genre_sequences, 0, self.num_genres - 1)
            
            genre_embeddings = self.genre_embed(genre_sequences)  # (batch, seq, max_genres, embed_size)
            
            # Aggregate genre embeddings
            aggregated_genre_emb = self._aggregate_genre_embeddings(genre_embeddings)  # (batch, seq, embed_size)
            
            # Fusion
            combined = torch.cat([token_emb, aggregated_genre_emb], dim=-1)
        else:
            # Single genre processing (original)
            genre_sequence = self._get_single_genre_mapping(sequence)
            
            # Ensure genre sequence is within bounds
            genre_sequence = torch.clamp(genre_sequence, 0, self.num_genres - 1)
            
            genre_emb = self.genre_embed(genre_sequence)
            combined = torch.cat([token_emb, genre_emb], dim=-1)
        
        # 3. Apply fusion layer
        x = self.fusion_layer(combined)
        
        return self.dropout(x)
    
    def get_anime_genre_info(self, token_id):
        """Get genre information for a specific anime token"""
        if not self.has_mappings:
            return None
        
        token_idx = torch.where(self.token_ids == token_id)[0]
        if len(token_idx) == 0:
            return None
        
        if self.multi_genre:
            return self.genre_ids[token_idx[0]].cpu().numpy().tolist()
        else:
            return [self.genre_ids[token_idx[0]].item()]
    
    def get_user_genre_distribution(self, user_sequence):
        """Analyze genre distribution in user's anime list"""
        genre_counts = {}
        
        for token_id in user_sequence:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
            
            genres = self.get_anime_genre_info(token_id)
            if genres:
                for genre_id in genres:
                    if genre_id != 0:  # Skip padding
                        genre_counts[genre_id] = genre_counts.get(genre_id, 0) + 1
        
        return genre_counts
    
    def clear_cache(self):
        """Clear internal caches to free GPU memory"""
        with self._cache_lock:
            self._genre_cache.clear()
    
    @classmethod
    def clear_global_cache(cls):
        """Clear global mappings cache"""
        with cls._cache_lock:
            cls._mappings_cache = None

    def get_top_genres(self, user_sequence, top_k=5):
        """Get user's top K preferred genres"""
        analysis = self.get_user_genre_distribution(user_sequence)
        
        sorted_genres = sorted(
            analysis.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_genres[:top_k]
