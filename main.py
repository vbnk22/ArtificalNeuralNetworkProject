import pandas as pd
import matplotlib.pyplot as plt
from keras.src.layers import Dropout
from keras.src.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

data = pd.read_csv('D:\\Python\\projektSieci\\data.csv')

# Uzupełnianie danych tekstowych
data['Weather'] = data['Weather'].fillna(data['Weather'].mode()[0])
data['Road_Type'] = data['Road_Type'].fillna(data['Road_Type'].mode()[0])
data['Time_of_Day'] = data['Time_of_Day'].fillna(data['Time_of_Day'].mode()[0])
data['Accident_Severity'] = data['Accident_Severity'].fillna(data['Accident_Severity'].mode()[0])
data['Road_Condition'] = data['Road_Condition'].fillna(data['Road_Condition'].mode()[0])
data['Vehicle_Type'] = data['Vehicle_Type'].fillna(data['Vehicle_Type'].mode()[0])
data['Road_Light_Condition'] = data['Road_Light_Condition'].fillna(data['Road_Light_Condition'].mode()[0])

# Uzupełnianie danych numerycznych
data['Traffic_Density'] = data['Traffic_Density'].fillna(data['Traffic_Density'].median())
data['Speed_Limit'] = data['Speed_Limit'].fillna(data['Speed_Limit'].median())
data['Number_of_Vehicles'] = data['Number_of_Vehicles'].fillna(data['Number_of_Vehicles'].median())
data['Driver_Alcohol'] = data['Driver_Alcohol'].fillna(data['Driver_Alcohol'].median())
data['Driver_Age'] = data['Driver_Age'].fillna(data['Driver_Age'].median())
data['Driver_Experience'] = data['Driver_Experience'].fillna(data['Driver_Experience'].median())
data['Accident'] = data['Accident'].fillna(data['Accident'].median())

# Usunięcie duplikatów
data = data.drop_duplicates()

# 'Accident' jest zmienną celu (target)
X = data.drop(columns=['Accident'])  # Cechy (features)
y = data['Accident']  # Zmienna celu (target)

# Kodowanie danych kategorycznych (np. One-Hot Encoding dla zmiennych tekstowych)
X = pd.get_dummies(X)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Skalowanie danych (standardyzacja)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Budowanie modelu
model = Sequential()

# Dodanie warstwy wejściowej i pierwszej warstwy ukrytej
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Dodanie drugiej warstwy ukrytej
model.add(Dense(64, activation='relu'))
# Regularyzacja l2 i Dropout
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.8))

# Warstwa wyjściowa (1 neuron, bo mamy tylko 2 klasy: 0 i 1)
model.add(Dense(1, activation='sigmoid'))

# Kompilowanie modelu
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Ocena modelu na zbiorze testowym
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Wizualizacja wyników treningu
plt.plot(history.history['accuracy'], label='Dokładność (trening)')
plt.plot(history.history['val_accuracy'], label='Dokładność (walidacja)')
plt.title('Dokładność modelu w czasie')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()
plt.show()

# Wizualizacja funkcji straty
plt.plot(history.history['loss'], label='Strata (trening)')
plt.plot(history.history['val_loss'], label='Strata (walidacja)')
plt.title('Funkcja straty w czasie')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()
plt.show()

# Predykcja na zbiorze testowym
y_pred = model.predict(X_test)

# Przekształcenie predykcji na prawdopodobieństwa wypadku
y_pred_prob = y_pred.flatten()  # Spłaszczenie tablicy predykcji
y_pred_binary = (y_pred_prob > 0.5).astype(int)  # Klasyfikacja (0 lub 1)

# Odtwarzanie oryginalnych kolumn z zakodowanych danych
decoded_X_test = pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns)

# Przekształcenie One-Hot Encoding z powrotem na oryginalne wartości
decoded_X_test['Weather'] = decoded_X_test[['Weather_Clear', 'Weather_Foggy', 'Weather_Rainy', 'Weather_Snowy', 'Weather_Stormy']].idxmax(axis=1).str.replace('Weather_', '')
decoded_X_test['Road_Type'] = decoded_X_test[['Road_Type_City Road', 'Road_Type_Highway', 'Road_Type_Mountain Road', 'Road_Type_Rural Road']].idxmax(axis=1).str.replace('Road_Type_', '')
decoded_X_test['Time_of_Day'] = decoded_X_test[['Time_of_Day_Afternoon', 'Time_of_Day_Evening', 'Time_of_Day_Morning', 'Time_of_Day_Night']].idxmax(axis=1).str.replace('Time_of_Day_', '')
decoded_X_test['Accident_Severity'] = decoded_X_test[['Accident_Severity_High', 'Weather_Foggy', 'Accident_Severity_Low', 'Accident_Severity_Moderate']].idxmax(axis=1).str.replace('Accident_Severity_', '')
decoded_X_test['Road_Condition'] = decoded_X_test[['Road_Condition_Dry', 'Road_Condition_Icy', 'Road_Condition_Under Construction', 'Road_Condition_Wet']].idxmax(axis=1).str.replace('Road_Condition_', '')
decoded_X_test['Vehicle_Type'] = decoded_X_test[['Vehicle_Type_Bus', 'Vehicle_Type_Car', 'Vehicle_Type_Motorcycle', 'Vehicle_Type_Truck']].idxmax(axis=1).str.replace('Vehicle_Type_', '')
decoded_X_test['Road_Light_Condition'] = decoded_X_test[['Road_Light_Condition_Artificial Light', 'Road_Light_Condition_Daylight', 'Road_Light_Condition_No Light']].idxmax(axis=1).str.replace('Road_Light_Condition_', '')

# Dodanie kolumn z predykcjami
decoded_X_test['Predicted_Accident_Risk'] = y_pred_prob  # Szansa na wypadek
decoded_X_test['Accident_Prediction'] = y_pred_binary   # Klasyfikacja (0 lub 1)
decoded_X_test['True_Accident'] = y_test.values         # Rzeczywiste dane celu

# Wyświetlanie wybranych kolumn
columns_to_display = ['Weather', 'Road_Type', 'Time_of_Day', 'Traffic_Density',
                      'Speed_Limit', 'Number_of_Vehicles', 'Driver_Alcohol', 'Accident_Severity', 'Road_Condition', 'Vehicle_Type', 'Driver_Age',
                      'Driver_Experience', 'Road_Light_Condition', 'Predicted_Accident_Risk', 'Accident_Prediction', 'True_Accident']
decoded_X_test[columns_to_display].to_csv('D:\\Python\\projektSieci\\out_data.csv', index=False)
