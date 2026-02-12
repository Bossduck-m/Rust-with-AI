use rust_with_my_ai::*;

fn main() {
    println!("🚀 Rust-with-my-AI: XOR Örneği Başlatılıyor...");

    // Model Yapılandırması
    let mut model = NeuralNetwork::new(vec![
        Box::new(DenseLayer::new(2, 4, Activation::ReLU)),
        Box::new(LayerNorm::new(4)),
        Box::new(Residual {
            layer: Box::new(DenseLayer::new(4, 4, Activation::ReLU)),
        }),
        Box::new(DenseLayer::new(4, 1, Activation::Sigmoid)),
    ]);

    // XOR Veri Seti
    let xor_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    // Eğitim
    let mut optimizer = Adam::new(0.01);
    println!("Eğitim başlıyor...");
    model.train(&xor_data, 1000, 4, &mut optimizer);

    // Test ve Tahmin
    println!("\n--- Tahmin Sonuçları ---");
    for (input_vec, target_vec) in &xor_data {
        let predict = model.predict(input_vec);
        println!(
            "Girdi: {:?}, Hedef: {:?}, Tahmin: {:.4}",
            input_vec, target_vec, predict[0]
        );
    }
    println!("\n✅ Örnek başarıyla tamamlandı.");
}
