#  ML Model Deployment API

A lightweight RESTful API for serving two machine learning models using Flask and Docker.

##  Models Included

1. **Iris Flower Classification**  
   Predict the species of an iris flower from its sepal/petal dimensions.

2. **House Price Prediction**  
   Estimate the price of a house based on structural and locational features.

---

## ️ Setup Instructions

### 1. Train and Save Models

```bash
python train.py            # Trains and saves model.pkl (Iris classifier)
python house_price.py      # Trains and saves house_price_model.pkl
```

### 2. Build Docker Image

```bash
docker build -t ml-model .
```

### 3. Run the Docker Container

```bash
docker run -p 9000:9000 ml-model
```

API will be available at: [http://localhost:9000](http://localhost:9000)

---

##  API Endpoints

### 🩺 Health Check

- **Method:** `GET`
- **Endpoint:** `/health`

**Response:**

```json
{
  "status": "OK!"
}
```

---

###  Iris Flower Classification

- **Method:** `POST`
- **Endpoint:** `/predict`

####  Request Body

Send either a single sample or multiple samples:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

or

```json
{
  "features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3]
  ]
}
```

####  Response

```json
{
    "predictions": [
        {
            "confidence": 1.0,
            "prediction": 0
        },
        {
            "confidence": 0.99,
            "prediction": 2
        }
    ]
}
```

**Prediction Classes:**

- `0`: Setosa  
- `1`: Versicolor  
- `2`: Virginica

---

###  House Price Prediction

- **Method:** `POST`
- **Endpoint:** `/predict_house`

####  Request Body

```json
{
  "features": [
    9960,
    3,
    2,
    2,
    "yes",
    "no",
    "yes",
    "no",
    "no",
    2,
    "yes",
    "semi-furnished"
  ]
}
```

####  Feature Descriptions (in order)

| Index | Feature            | Type       | Example           |
|-------|--------------------|------------|-------------------|
| 0     | area               | float      | 9960              |
| 1     | bedrooms           | int        | 3                 |
| 2     | bathrooms          | int        | 2                 |
| 3     | stories            | int        | 2                 |
| 4     | mainroad           | "yes"/"no" | "yes"             |
| 5     | guestroom          | "yes"/"no" | "no"              |
| 6     | basement           | "yes"/"no" | "yes"             |
| 7     | hotwaterheating    | "yes"/"no" | "no"              |
| 8     | airconditioning    | "yes"/"no" | "no"              |
| 9     | parking            | int        | 2                 |
| 10    | prefarea           | "yes"/"no" | "yes"             |
| 11    | furnishingstatus   | string     | "semi-furnished"  |

####  Response

```json
{
  "predicted_price": 1234567.89
}
```

---

##  Error Handling

| Code | Meaning                | When it Happens                          |
|------|------------------------|------------------------------------------|
| 400  | Bad Request            | Malformed or missing `features` field    |
| 500  | Internal Server Error  | Unhandled processing exception           |

**Example:**

```json
{
  "error": "Invalid Input: Features must be a list"
}
```

---

##  Example Usage

###  Iris Classification

```bash
curl -X POST http://localhost:9000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

###  House Price Prediction

```bash
curl -X POST http://localhost:9000/predict_house \
     -H "Content-Type: application/json" \
     -d '{
           "features": [
             9960, 3, 2, 2, "yes", "no", "yes",
             "no", "no", 2, "yes", "semi-furnished"
           ]
         }'
```

---

##  Project Structure

```
.
├── app.py                  # Flask API
├── train.py                # Iris model training
├── house_price.py          # House price model training
├── model.pkl               # Iris classifier model file
├── house_price_model.pkl   # House price model file
├── requirements.txt        # Python dependencies
└── Dockerfile              # Docker container definition
```

---

##  License

This project is licensed under the **MIT License**.
