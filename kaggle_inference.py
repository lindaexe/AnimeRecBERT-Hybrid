import json
import pickle
import torch
import numpy as np
from pathlib import Path
import re
import random

class AnimeRecommendationSystem:
    def __init__(self, checkpoint_path, dataset_path, animes_path, images_path=None, 
                 mal_urls_path=None, type_seq_path=None, genres_path=None):
        self.model = None
        self.dataset = None
        self.id_to_anime = {}
        self.id_to_url = {}
        self.id_to_mal_url = {}
        self.id_to_genres = {}
        self.id_to_type_seq = {}
        
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.animes_path = animes_path
        self.images_path = images_path
        self.mal_urls_path = mal_urls_path
        self.type_seq_path = type_seq_path
        self.genres_path = genres_path
        
        self.favorite_animes = []
        self.blacklisted_animes = []
        
        self.load_model_and_data()

    def load_model_and_data(self):
        try:
            print("Loading model and data...")
            
            # Dataset yükleme
            dataset_path = Path(self.dataset_path)
            with dataset_path.open('rb') as f:
                self.dataset = pickle.load(f)["smap"]
            
            # Anime isimleri yükleme
            with open(self.animes_path, "r", encoding="utf-8") as file:
                self.id_to_anime = json.load(file)
            
            # Opsiyonel dosyaları yükleme
            if self.images_path and Path(self.images_path).exists():
                with open(self.images_path, "r", encoding="utf-8") as file:
                    self.id_to_url = json.load(file)
                print(f"Loaded {len(self.id_to_url)} image URLs")
            
            if self.mal_urls_path and Path(self.mal_urls_path).exists():
                with open(self.mal_urls_path, "r", encoding="utf-8") as file:
                    self.id_to_mal_url = json.load(file)
                print(f"Loaded {len(self.id_to_mal_url)} MAL URLs")
            
            if self.type_seq_path and Path(self.type_seq_path).exists():
                with open(self.type_seq_path, "r", encoding="utf-8") as file:
                    self.id_to_type_seq = json.load(file)
                print(f"Loaded {len(self.id_to_type_seq)} type/sequel info")
            
            if self.genres_path and Path(self.genres_path).exists():
                with open(self.genres_path, "r", encoding="utf-8") as file:
                    self.id_to_genres = json.load(file)
                print(f"Loaded {len(self.id_to_genres)} genre info")
            
            # Model yükleme - Bu kısmı gerçek model yapınıza göre uyarlayın
            self.load_checkpoint()
            
            print("Model loaded successfully!")
            print(f"Total animes in dataset: {len(self.id_to_anime)}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

    def load_checkpoint(self):
        try:
            # Bu kısmı gerçek model yapınıza göre uyarlayın
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = torch.load(f, map_location='cpu', weights_only=False)
            
            # Model yükleme kısmını gerçek model sınıfınıza göre uyarlayın
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.model.eval()
            
            print("Checkpoint loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {str(e)}")
            # Model olmadan da çalışabilir (rastgele öneriler için)
            self.model = None

    def search_anime(self, query):
        """Anime arama fonksiyonu"""
        query = query.lower().strip()
        matches = []
        
        for anime_id, anime_data in self.id_to_anime.items():
            anime_names = anime_data if isinstance(anime_data, list) else [anime_data]
            
            for name in anime_names:
                if query in name.lower():
                    main_name = anime_names[0] if anime_names else "Unknown"
                    matches.append({
                        'id': int(anime_id),
                        'name': main_name,
                        'score': len(query) / len(name)  # Basit skorlama
                    })
                    break
        
        # Skorlara göre sırala
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:20]  # En iyi 20 sonucu döndür

    def get_anime_details(self, anime_id):
        """Anime detaylarını getir"""
        anime_id_str = str(anime_id)
        if anime_id_str not in self.id_to_anime:
            return None
        
        anime_data = self.id_to_anime[anime_id_str]
        anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)
        
        details = {
            'id': anime_id,
            'name': anime_name,
            'image_url': self.id_to_url.get(anime_id_str),
            'mal_url': self.id_to_mal_url.get(anime_id_str),
            'genres': self.get_anime_genres(anime_id)
        }
        
        return details

    def get_anime_genres(self, anime_id):
        """Anime türlerini getir"""
        genres = self.id_to_genres.get(str(anime_id), [])
        return [genre for genre in genres[0]] if genres else []

    def add_favorite(self, anime_id):
        """Favori anime ekle"""
        if anime_id not in self.favorite_animes:
            self.favorite_animes.append(anime_id)
            return True
        return False

    def remove_favorite(self, anime_id):
        """Favori animeden çıkar"""
        if anime_id in self.favorite_animes:
            self.favorite_animes.remove(anime_id)
            return True
        return False

    def add_blacklist(self, anime_id):
        """Kara listeye ekle"""
        if anime_id not in self.blacklisted_animes:
            self.blacklisted_animes.append(anime_id)
            return True
        return False

    def remove_blacklist(self, anime_id):
        """Kara listeden çıkar"""
        if anime_id in self.blacklisted_animes:
            self.blacklisted_animes.remove(anime_id)
            return True
        return False

    def get_recommendations(self, num_recommendations=20):
        """Anime önerileri getir"""
        if not self.favorite_animes:
            return [], "Please add some favorite animes first!"
        
        if self.model is None:
            # Model yoksa rastgele öneriler yap
            return self._get_random_recommendations(num_recommendations)
        
        try:
            smap = self.dataset
            inverted_smap = {v: k for k, v in smap.items()}
            
            converted_ids = []
            for anime_id in self.favorite_animes:
                if anime_id in smap:
                    converted_ids.append(smap[anime_id])
            
            if not converted_ids:
                return [], "None of the selected animes are in the model vocabulary!"
            
            # Model tahminleri
            target_len = 128
            padded = converted_ids + [0] * (target_len - len(converted_ids))
            input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)
            
            max_predictions = min(500, len(inverted_smap))
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                last_logits = logits[:, -1, :]
                top_scores, top_indices = torch.topk(last_logits, k=max_predictions, dim=1)
            
            recommendations = []
            
            for idx, score in zip(top_indices.numpy()[0], top_scores.detach().numpy()[0]):
                if idx in inverted_smap:
                    anime_id = inverted_smap[idx]
                    
                    # Favori ve kara listede olanları atla
                    if anime_id in self.favorite_animes or anime_id in self.blacklisted_animes:
                        continue
                    
                    if str(anime_id) in self.id_to_anime:
                        details = self.get_anime_details(anime_id)
                        if details:
                            details['score'] = float(score)
                            recommendations.append(details)
                        
                        if len(recommendations) >= num_recommendations:
                            break
            
            return recommendations, f"Found {len(recommendations)} recommendations!"
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return self._get_random_recommendations(num_recommendations)

    def _get_random_recommendations(self, num_recommendations):
        """Rastgele öneriler (model yokken)"""
        all_anime_ids = [int(k) for k in self.id_to_anime.keys()]
        available_ids = [aid for aid in all_anime_ids 
                        if aid not in self.favorite_animes and aid not in self.blacklisted_animes]
        
        if not available_ids:
            return [], "No available animes for recommendation!"
        
        selected_ids = random.sample(available_ids, min(num_recommendations, len(available_ids)))
        
        recommendations = []
        for anime_id in selected_ids:
            details = self.get_anime_details(anime_id)
            if details:
                details['score'] = random.random()  # Rastgele skor
                recommendations.append(details)
        
        return recommendations, f"Found {len(recommendations)} random recommendations!"

    def print_favorites(self):
        """Favori animeleri yazdır"""
        if not self.favorite_animes:
            print("No favorite animes added yet.")
            return
        
        print("\n=== FAVORITE ANIMES ===")
        for i, anime_id in enumerate(self.favorite_animes, 1):
            details = self.get_anime_details(anime_id)
            if details:
                print(f"{i}. {details['name']} (ID: {anime_id})")
                if details['genres']:
                    print(f"   Genres: {', '.join(details['genres'])}")

    def print_blacklist(self):
        """Kara listeyi yazdır"""
        if not self.blacklisted_animes:
            print("No blacklisted animes.")
            return
        
        print("\n=== BLACKLISTED ANIMES ===")
        for i, anime_id in enumerate(self.blacklisted_animes, 1):
            details = self.get_anime_details(anime_id)
            if details:
                print(f"{i}. {details['name']} (ID: {anime_id})")


def main():
    checkpoint_path = "Data/AnimeRatings/pretrained_bert.pth"
    dataset_path = "Data/AnimeRatings/dataset.pkl"
    animes_path = "Data/animes.json"
    images_path = "Data/id_to_url.json" 
    mal_urls_path = "Data/anime_to_malurl.json" 
    type_seq_path = "Data/anime_to_typenseq.json"
    genres_path = "Data/id_to_genres.json" 
    
    try:
        # Sistem başlatma
        print("Initializing Anime Recommendation System...")
        recommender = AnimeRecommendationSystem(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            animes_path=animes_path,
            images_path=images_path,
            mal_urls_path=mal_urls_path,
            type_seq_path=type_seq_path,
            genres_path=genres_path
        )
        
        print("\n" + "="*60)
        print("🎌 ANIME RECOMMENDATION SYSTEM 🎌")
        print("="*60)
        
        while True:
            print("\n--- MENU ---")
            print("1. Search and add anime to favorites")
            print("2. Remove anime from favorites")
            print("3. Add anime to blacklist")
            print("4. Remove anime from blacklist")
            print("5. View favorites")
            print("6. View blacklist")
            print("7. Get recommendations")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == "1":
                # Anime arama ve ekleme
                query = input("Enter anime name to search: ").strip()
                if query:
                    matches = recommender.search_anime(query)
                    if matches:
                        print(f"\nFound {len(matches)} matches:")
                        for i, match in enumerate(matches, 1):
                            print(f"{i}. {match['name']} (ID: {match['id']})")
                        
                        try:
                            choice_idx = int(input("Select anime number (0 to cancel): ")) - 1
                            if 0 <= choice_idx < len(matches):
                                selected_anime = matches[choice_idx]
                                if recommender.add_favorite(selected_anime['id']):
                                    print(f"✅ Added '{selected_anime['name']}' to favorites!")
                                else:
                                    print("❌ Anime already in favorites!")
                            elif choice_idx != -1:
                                print("❌ Invalid selection!")
                        except ValueError:
                            print("❌ Please enter a valid number!")
                    else:
                        print("❌ No matches found!")
            
            elif choice == "2":
                # Favori animeden çıkarma
                recommender.print_favorites()
                if recommender.favorite_animes:
                    try:
                        anime_id = int(input("Enter anime ID to remove from favorites: "))
                        if recommender.remove_favorite(anime_id):
                            print("✅ Removed from favorites!")
                        else:
                            print("❌ Anime not found in favorites!")
                    except ValueError:
                        print("❌ Please enter a valid anime ID!")
            
            elif choice == "3":
                # Kara listeye ekleme
                query = input("Enter anime name to search and blacklist: ").strip()
                if query:
                    matches = recommender.search_anime(query)
                    if matches:
                        print(f"\nFound {len(matches)} matches:")
                        for i, match in enumerate(matches, 1):
                            print(f"{i}. {match['name']} (ID: {match['id']})")
                        
                        try:
                            choice_idx = int(input("Select anime number (0 to cancel): ")) - 1
                            if 0 <= choice_idx < len(matches):
                                selected_anime = matches[choice_idx]
                                if recommender.add_blacklist(selected_anime['id']):
                                    print(f"✅ Added '{selected_anime['name']}' to blacklist!")
                                else:
                                    print("❌ Anime already in blacklist!")
                            elif choice_idx != -1:
                                print("❌ Invalid selection!")
                        except ValueError:
                            print("❌ Please enter a valid number!")
                    else:
                        print("❌ No matches found!")
            
            elif choice == "4":
                # Kara listeden çıkarma
                recommender.print_blacklist()
                if recommender.blacklisted_animes:
                    try:
                        anime_id = int(input("Enter anime ID to remove from blacklist: "))
                        if recommender.remove_blacklist(anime_id):
                            print("✅ Removed from blacklist!")
                        else:
                            print("❌ Anime not found in blacklist!")
                    except ValueError:
                        print("❌ Please enter a valid anime ID!")
            
            elif choice == "5":
                # Favorileri göster
                recommender.print_favorites()
            
            elif choice == "6":
                # Kara listeyi göster
                recommender.print_blacklist()
            
            elif choice == "7":
                # Önerileri getir
                if not recommender.favorite_animes:
                    print("❌ Please add some favorite animes first!")
                    continue
                
                try:
                    num_rec = int(input("Number of recommendations (default 20): ") or "20")
                    recommendations, message = recommender.get_recommendations(num_rec)
                    
                    print(f"\n{message}")
                    if recommendations:
                        print("\n🎯 RECOMMENDATIONS:")
                        print("="*50)
                        for i, rec in enumerate(recommendations, 1):
                            print(f"{i}. {rec['name']} (ID: {rec['id']})")
                            if 'score' in rec:
                                print(f"   Score: {rec['score']:.4f}")
                            if rec.get('genres'):
                                print(f"   Genres: {', '.join(rec['genres'])}")
                            if rec.get('mal_url'):
                                print(f"   MAL: {rec['mal_url']}")
                            print()
                except ValueError:
                    print("❌ Please enter a valid number!")
            
            elif choice == "8":
                print("👋 Goodbye! Thanks for using the Anime Recommendation System!")
                break
            
            else:
                print("❌ Invalid choice! Please select 1-8.")
    
    except Exception as e:
        print(f"❌ Error initializing system: {str(e)}")
        print("Please check your file paths and data files.")

if __name__ == "__main__":
    main()
