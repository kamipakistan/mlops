from pydantic import BaseModel
from fastapi import FastAPI
from src.pipeline.predict_pipeline import PredictPipeline, CustomData


app = FastAPI()

class StudentFeatures(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int


@app.get('/')
async def root():
    return {'message': 'Student Math Score Prediction'}


@app.post('/predict')
async def predict(data: StudentFeatures):
    custom_data = CustomData(
        gender=data.gender,
        race_ethnicity=data.race_ethnicity,
        parental_level_of_education=data.parental_level_of_education,
        lunch=data.lunch,
        test_preparation_course=data.test_preparation_course,
        reading_score=data.reading_score,
        writing_score=data.writing_score
    )

    df = custom_data.get_data_as_data_frame()
    prediction = PredictPipeline().predict(df)

    return {'Prediction': float(prediction[0])}

