# **VAE-Based AI Nutrition Recommendation System**
*A Deep Generative Model for Personalized Diet Planning*

## **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Results](#results)
- [Limitations & Future Work](#limitations--future-work)
- [Contributing](#contributing)
- [License](#license)

---

## **Introduction**
This project implements an **AI-based nutrition recommendation system** using a **Variational Autoencoder (VAE)** combined with **ChatGPT** for **personalized meal planning**. It leverages deep generative networks to create weekly meal plans aligned with established nutritional guidelines, ensuring energy intake and macronutrient accuracy.

### **Why this project?**
- **Personalized**: Tailors meal plans to user profiles (age, weight, health conditions, etc.).
- **Explainable AI**: Uses a latent space to group users with similar nutritional needs.
- **Scalable**: Incorporates **ChatGPT** to expand meal variety.
- **Health-Oriented**: Ensures compliance with WHO & EFSA nutritional guidelines.

---

## **Features**
✔ Personalized meal plans based on anthropometric and medical data  
✔ Weekly meal generation with **six meals per day**  
✔ **Variational Autoencoder (VAE)** for user feature extraction  
✔ **Gated Recurrent Unit (GRU)-based decoder** for meal sequence generation  
✔ **Optimizer module** to adjust meal portions to match energy needs  
✔ Integration with **ChatGPT** to generate meal equivalents from diverse cuisines  
✔ **Validation on 3000 virtual profiles and 1000 real user profiles**  


## **Model Architecture**
The system comprises three main components:

1. **Variational Autoencoder (VAE)**  
   - Encodes user profiles into a **latent space** for better clustering.
   - Ensures similar users receive similar diet recommendations.

2. **GRU-Based Meal Generator**  
   - Generates daily meal plans as sequences of six meals (Breakfast to Supper).
   - Uses a **recurrent neural network** to maintain consistency.

3. **Meal Optimizer**  
   - Adjusts meal portion sizes to meet **energy intake** and **nutritional balance**.
