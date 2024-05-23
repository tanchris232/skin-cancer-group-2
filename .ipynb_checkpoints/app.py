{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "c0230c34",
   "metadata": {
    "scrolled": True
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: keras in /Users/raphael/anaconda3/lib/python3.11/site-packages (3.3.3)\n",
      "Requirement already satisfied: streamlit in /Users/raphael/anaconda3/lib/python3.11/site-packages (1.34.0)\n",
      "Requirement already satisfied: pillow in /Users/raphael/anaconda3/lib/python3.11/site-packages (9.4.0)\n",
      "Requirement already satisfied: numpy in /Users/raphael/anaconda3/lib/python3.11/site-packages (1.24.3)\n",
      "Requirement already satisfied: absl-py in /Users/raphael/anaconda3/lib/python3.11/site-packages (from keras) (2.1.0)\n",
      "Requirement already satisfied: rich in /Users/raphael/anaconda3/lib/python3.11/site-packages (from keras) (13.7.1)\n",
      "Requirement already satisfied: namex in /Users/raphael/anaconda3/lib/python3.11/site-packages (from keras) (0.0.7)\n",
      "Requirement already satisfied: h5py in /Users/raphael/anaconda3/lib/python3.11/site-packages (from keras) (3.10.0)\n",
      "Requirement already satisfied: optree in /Users/raphael/anaconda3/lib/python3.11/site-packages (from keras) (0.11.0)\n",
      "Requirement already satisfied: ml-dtypes in /Users/raphael/anaconda3/lib/python3.11/site-packages (from keras) (0.3.2)\n",
      "Requirement already satisfied: altair<6,>=4.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (5.3.0)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (1.8.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (5.3.3)\n",
      "Requirement already satisfied: click<9,>=7.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (8.0.4)\n",
      "Requirement already satisfied: packaging<25,>=16.8 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (23.2)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (2.0.3)\n",
      "Requirement already satisfied: protobuf<5,>=3.20 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (4.25.3)\n",
      "Requirement already satisfied: pyarrow>=7.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (11.0.0)\n",
      "Requirement already satisfied: requests<3,>=2.27 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (2.31.0)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (8.2.2)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (4.11.0)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (3.1.43)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (0.9.1)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from streamlit) (6.3.2)\n",
      "Requirement already satisfied: jinja2 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from altair<6,>=4.0->streamlit) (3.1.2)\n",
      "Requirement already satisfied: jsonschema>=3.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from altair<6,>=4.0->streamlit) (4.17.3)\n",
      "Requirement already satisfied: toolz in /Users/raphael/anaconda3/lib/python3.11/site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3.post1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.27->streamlit) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.27->streamlit) (2.2.1)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.27->streamlit) (2023.7.22)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from rich->keras) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from rich->keras) (2.15.1)\n",
      "Requirement already satisfied: smmap<6,>=3.0.1 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.1)\n",
      "Requirement already satisfied: attrs>=17.4.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (22.1.0)\n",
      "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.18.0)\n",
      "Requirement already satisfied: mdurl~=0.1 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from markdown-it-py>=2.2.0->rich->keras) (0.1.0)\n",
      "Requirement already satisfied: six>=1.5 in /Users/raphael/anaconda3/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install keras streamlit pillow numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "d5eec986",
   "metadata": {
    "scrolled": True
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "from PIL import Image, ImageOps\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "\n",
    "# Load your pre-trained model\n",
    "@st.cache_resource  # Cache the model loading so it doesn't reload every time the script runs\n",
    "def load_model():\n",
    "    model = tf.keras.models.load_model('/Users/raphael/Documents/GitHub/skin-cancer-group-2/skin_cancer_classifier_test.h5')\n",
    "    return model\n",
    "\n",
    "model = load_model()\n",
    "\n",
    "# Function to preprocess the image and make predictions\n",
    "def preprocess_image(image):\n",
    "    size = (224, 224)  # The size your model expects\n",
    "    image = ImageOps.fit(image, size, Image.ANTIALIAS)\n",
    "    image_array = np.asarray(image)\n",
    "    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1\n",
    "    data = np.expand_dims(normalized_image_array, axis=0)\n",
    "    return data\n",
    "\n",
    "def predict_image(image):\n",
    "    processed_image = preprocess_image(image)\n",
    "    prediction = model.predict(processed_image)\n",
    "    return \"Malignant\" if prediction[0][0] > 0.5 else \"Benign\"\n",
    "\n",
    "# Streamlit app layout\n",
    "st.title(\"Mole Type Classification\")\n",
    "st.write(\"Please upload an image to determine whether it is classified as benign or malignant.\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Please choose an image...\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    image = Image.open(uploaded_file)\n",
    "    st.image(image, caption=\"Uploaded Image.\", use_column_width=True)\n",
    "    st.write(\"\")\n",
    "    st.write(\"Classifying...\")\n",
    "\n",
    "    label = predict_image(image)\n",
    "    st.write(f'Prediction: {label}')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "745686e5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
