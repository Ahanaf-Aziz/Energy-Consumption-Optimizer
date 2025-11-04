# Energy Consumption Optimization 🔋

Ever wonder how much energy buildings actually waste? This project tackles that problem by using machine learning to predict and optimize energy usage based on weather patterns and real consumption data.

## What This Project Does 🎯

We built a system that combines several components:
- ⚡ Energy meter readings from buildings
- 🌡️ Weather data like temperature and humidity
- 🤖 Two different ML models (Random Forest and LSTM) to see which works better
- 💡 Practical strategies to cut down energy waste

## What You Get ✨

The project includes everything from start to finish:
- 🧹 Data cleaning and feature engineering to prepare the raw data
- 🌲 A Random Forest model for quick predictions
- 🧠 An LSTM deep learning model for more complex patterns
- 📊 Side-by-side comparison of how well each model performs (using RMSE, R², and MAE)
- 💰 Five actionable strategies to save energy, with actual numbers showing how much you'd save
- 📈 Graphs and detailed analysis of the results

## Our Results 🎉

Here's what we found:
- Random Forest gave us an error of about 4.5 kWh
- LSTM performed better at 3.8 kWh
- We could cut annual energy consumption by roughly 42%
- The biggest win came from managing peak demand times (15% reduction)

## The Code 💻

The project is split into five easy-to-follow scripts:

1. `1_generate_and_preprocess_data.py` — Creates test data and gets it ready for modeling
2. `2_train_random_forest_model.py` — Builds the Random Forest model
3. `3_train_lstm_model.py` — Builds the LSTM deep learning model
4. `4_energy_optimization_strategies.py` — Finds where we can actually save energy
5. `5_evaluate_and_visualize.py` — Compares the models and makes charts

## Getting Started 🚀

### Google Colab (Best for Getting Started Quickly)
1. Head to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Start a new notebook
3. Follow the codes on repository 
4. Copy and paste each script into separate cells
5. Run in order (1 → 2 → 3 → 4 → 5)
<img width="1389" height="985" alt="image" src="https://github.com/user-attachments/assets/72b6bc5f-80b7-4cb5-bc98-cb00781fbb3a" />
