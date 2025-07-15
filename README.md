# AnimeRecBERT: BERT-Based Anime Recommendation System

**AnimeRecBERT** is a personalized anime recommendation system based on BERT transformer architecture. Adapted from [https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch), this project introduces customizations tailored for an anime recommendation system and inference.

- ⚙️ **Hybrid Model with Genre Embeddings** — Added genre-based embeddings to enrich BERT inputs with auxiliary item metadata.
- ⚙️ **No Positional Encoding** — Removed positional encoding due to lack of temporal signals in the dataset, which improved performance.
- 🎌 **Anime-Specific User-Item Dataset** — Built on a large-scale dataset tailored for anime recommendations.
- 🖥️ **GUI Interface** — Interactive interface for real-time recommendation visualization.
- 🌐 **Web Demo** — Live demonstration available in the browser.

This project provides a solid foundation for further development in personalized anime recommendation using transformer-based models.

## Metrics
The model was trained on a large-scale dataset containing 1.77 million users and 148 million ratings. Although positional encoding was removed, the results remain very close to those of the original BERT4Rec repository.
Below are the Top-K recommendation metrics:

<table>
<tr>
<td>

| Metric       | Value     |
|--------------|-----------|
| Recall@100   | 0.99996   |
| NDCG@100     | 0.7811    |
| Recall@50    | 0.9976    |
| NDCG@50      | 0.7807    |
| Recall@20    | 0.9863    |
| NDCG@20      | 0.7784    |
| Recall@10    | 0.9593    |
| NDCG@10      | 0.7714    |
| Recall@5     | 0.8996    |
| NDCG@5       | 0.7518    |
| Recall@1     | 0.5716    |
| NDCG@1       | 0.5716    |

</td>
</tr>
</table>

## Setup & Usage
### Clone Repo

```bash
git clone https://github.com/MRamazan/AnimeRecBERT
cd AnimeRecBERT
```

###  Create & Activate venv
#### For Linux
```
python3 -m venv venv
source venv/bin/activate 
```

#### For Windows
```
python -m venv venv
venv\Scripts\activate 
```


### Download Dataset & Pretrained Model

#### For Linux
```bash
curl -L -o Data/AnimeRatings/animeratings.zip \
     https://www.kaggle.com/api/v1/datasets/download/tavuksuzdurum/user-animelist-dataset

unzip Data/AnimeRatings/animeratings.zip -d Data/AnimeRatings/
```

#### For Windows
```bash
kaggle datasets download -d tavuksuzdurum/user-animelist-dataset -p Data\AnimeRatings

Expand-Archive -Path "Data\AnimeRatings\user-animelist-dataset.zip" -DestinationPath "Data\AnimeRatings" -Force
```

### Install Requirements
Install PyTorch from https://pytorch.org/get-started/locally/
```bash
pip install -r requirements.txt
```

### Run Local Host

```bash
python main_local.py \
    --checkpoint-path Data/AnimeRatings/best_acc_model.pth \
    --dataset-path Data/AnimeRatings/dataset.pkl \
    --animes-path Data/animes.json \
    --images-path Data/id_to_url.json \
    --mal-urls-path Data/anime_to_malurl.json \
    --type-seq-path Data/anime_to_typenseq.json \
    --genres-path Data/id_to_genres.json \     
```

### Train Code (**Not Required for inference**)
you can set parameters in templates.py file
```bash
# This script will train, validate and test the model.
# Training not required for inference.
python main.py  --template train_bert             
```

### Web GUI
<img src="gui.png" alt="BERTRec GUI" width="900">

# Results
## 🌟 My Favorites (Input for Inference)

| #  | Anime Title                                                                |
|----|----------------------------------------------------------------------------|
| 1  | Youkoso Jitsuryoku Shijou Shugi no Kyoushitsu e                            |
| 2  | Giji Harem                                                                 |
| 3  | Ijiranaide, Nagatoro-san                                                   |
| 4  | 86 (Eighty-Six)                                                            |
| 5  | Mushoku Tensei: Isekai Ittara Honki Dasu                                   |
| 6  | Made in Abyss                                                              |
| 7  | Shangri-La Frontier: Kusoge Hunter, Kamige ni Idoman to su                 |
| 8  | Vanitas no Karte                                                           |
| 9  | Jigokuraku                                                                 |

## 🌟 Recommendations Based on My Favorites
**Note:** The *position of favorites does not affect inference results*, as the model uses only the presence of items (not sequence).

## 🏆 Top Anime Recommendations for Me

| Rank | Anime Title                                                               
|------|------------------------------------------------------------------------------
| #1   | Yofukashi no Uta                                                      |
| #2   | Summertime Render                                                     |
| #3   | Mushoku Tensei II: Isekai Ittara Honki Dasu                           | 
| #4   | Tengoku Daimakyou                                                     | 
| #5   | Jujutsu Kaisen                                                        |
| #6   | Horimiya                                                              |       
| #7   | Chainsaw Man                                                          |     
| #8   | 86 Part 2                                                             |      
| #9   | Mushoku Tensei: Isekai Ittara Honki Dasu Part 2                       |    
| #10  | Ore dake Level Up na Ken (Solo Leveling)                              |   
| #11  | Kage no Jitsuryokusha ni Naritakute! 2nd Season                       |
| #12  | Youkoso Jitsuryoku Shijou Shugi no Kyoushitsu e 2nd Season            |    
| #13  | Sousou no Frieren                                                     |    
| #14  | Tonikaku Kawaii (Tonikawa: Over the Moon for You)                     |     
| #15  | Cyberpunk: Edgerunners                                                |   
| #16  | Tenki no Ko (Weathering With You)                                     |   
| #17  | Dandadan                                                              |      
| #18  | Spy x Family                                                          |
| #19  | Make Heroine ga Oosugiru!                                             |
| #20  | Boku no Kokoro no Yabai Yatsu                                         |



### ✅ Evaluation: How Good Are the Recommendations?

Out of the Top 20 recommendations, **10 titles** were already in my completed/favorites list — showing strong personalization performance.


| Watched & Liked? ✅ | Title                                                                  |
|---------------------|------------------------------------------------------------------------|
| ✅                  | Mushoku Tensei II: Isekai Ittara Honki Dasu                            |
| ✅                  | Mushoku Tensei: Isekai Ittara Honki Dasu Part 2                        |
| ✅                  | Youkoso Jitsuryoku Shijou Shugi no Kyoushitsu e 2nd Season             |
| ✅                  | Make Heroine ga Oosugiru!                                              |
| ✅                  | Spy x Family                                                           |
| ✅                  | Dandadan                                                               |
| ✅                  | 86 Part 2                                                              |
| ✅                  | Jujutsu Kaisen                                                         |
| ✅                  | Chainsaw Man                                                           |
| ✅                  | Cyberpunk: Edgerunners                                                 |

*I’m genuinely excited to watch the remaining anime as well — even with a quick glance, it’s clear they’re a great fit for my taste.*


