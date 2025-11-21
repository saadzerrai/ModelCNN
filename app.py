import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Configuration de la page
st.set_page_config(page_title="Reconnaissance d'Images CIFAR-10", page_icon="📷")

st.title("📷 Classificateur d'Images IA")
st.write("Téléchargez une image (Avion, Chat, Chien, etc.) et l'IA devinera ce que c'est !")


# 2. Chargement du modèle (avec cache pour ne pas le recharger à chaque clic)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cifar10_cnn_ameliore.h5')


try:
    model = load_model()
    st.success("Modèle chargé avec succès !")
except:
    st.error("Erreur : Le fichier 'mon_modele_cifar10.h5' est introuvable.")
    st.stop()

# Les classes du CIFAR-10
class_names = ['Avion ✈️', 'Automobile 🚗', 'Oiseau 🐦', 'Chat 🐱', 'Cerf 🦌',
               'Chien 🐶', 'Grenouille 🐸', 'Cheval 🐴', 'Bateau 🚢', 'Camion 🚛']

# 3. Interface d'upload
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Afficher l'image originale
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)

    st.write("🔍 Analyse en cours...")

    # 4. Prétraitement de l'image pour le modèle
    # Le modèle attend une image de taille (32, 32)
    img_resized = image.resize((32, 32))

    # Convertir en tableau numpy et normaliser (comme à l'entraînement)
    img_array = np.array(img_resized) / 255.0

    # Si l'image a 4 canaux (PNG transparent), on garde que les 3 premiers (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    # Ajouter la dimension du batch : (32, 32, 3) -> (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 5. Prédiction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])  # Convertir en probabilités

    class_index = np.argmax(predictions[0])
    confidence = 100 * np.max(score)  # Pourcentage de confiance

    # 6. Affichage du résultat
    st.markdown("---")
    st.header(f"C'est un(e) : **{class_names[class_index]}**")

    # Barre de progression pour la confiance
    st.progress(int(confidence))
    st.write(f"Confiance de l'IA : {confidence:.2f}%")