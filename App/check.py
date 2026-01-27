from tensorflow.keras.models import load_model

# Ganti path sesuai lokasi modelmu
model = load_model("base_model.h5")

print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
