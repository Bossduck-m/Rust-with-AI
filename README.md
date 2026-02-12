# 🧠 Rust-with-my-AI

Bu proje, Rust dilinde sıfırdan (sıfır bağımlılık prensibiyle) inşa edilmiş bir Sinir Ağı kütüphanesidir. Hem eğitim amaçlıdır hem de güçlü bir otograd (autograd) sistemi ile modern mimarileri (CNN, Transformers) destekleyecek şekilde tasarlanmıştır.

## 🚀 Özellikler

*   **Autograd Motoru**: Dinamik grafik tabanlı otomatik gradyan hesaplama.
*   **Transformer Blokları**: Mini-GPT mimarisi için `MultiHeadAttention`, `PositionalEncoding` ve `Causal Masking`.
*   **CNN Bileşenleri**: `Conv2D` (bias destekli), `MaxPooling` (tam backward destekli) ve `Flatten`.
*   **Modern Katmanlar**: `LayerNorm`, `Residual` bağlantılar ve `Dropout`.
*   **Optimizasyon**: `Adam` ve `SGD` optimizatörleri.
*   **Aktivasyonlar**: `ReLU`, `Sigmoid`, `Softmax` (Auto-axis support) ve `Linear`.

## 🎭 TinyShakespeareGPT

Proje içerisinde yer alan `TinyShakespeareGPT` modeli, karakter seviyesinde eğitilerek Shakespeare tarzında metin üretebilen otoregresif bir Transformer modelidir.

### Örnek Üretim:
> **ROMEO:** Ohevehsthat unsatiathalatisis de death

## 🛠️ Kurulum ve Test

Projeyi klonladıktan sonra Rust araç zinciriyle (Cargo) hemen test edebilirsiniz:

```bash
# GPT Modelini Test Et (Eğitim ve Üretim)
cargo test test_tiny_shakespeare_gpt -- --nocapture

# CNN (MNIST) Yapısını Test Et
cargo test test_mnist_cnn -- --nocapture

# Genel Bileşenleri Test Et
cargo test test_modern_components -- --nocapture
```

## 🏗️ Mimari Yaklaşım

Kütüphane, **"Feature-First" (d_model, seq_len)** konvansiyonuna göre optimize edilmiştir. Bu sayede matris işlemleri Rust'ın hızını en verimli şekilde kullanır.

---
*Bu proje, yapay zekanın "kalbine" inmek isteyenler için sıfırdan, her hücresi kontrol edilebilir bir altyapı sunar.*
