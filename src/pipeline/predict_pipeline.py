import sys
import pandas as pd
from src.exceptions import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path= model_path)
            preprocessor = load_object(file_path= preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        age:float, roomService:float, foodCourt:float, shoppingMall:float, spa:float, vRDeck:float,
        homePlanet:str, cryoSleep:str, destination:str, vIP:str, cabin:str):

        self.Age =age 
        self.RoomService=roomService 
        self.FoodCourt=foodCourt 
        self.ShoppingMall=shoppingMall
        self.Spa=spa
        self.VRDeck=vRDeck
        self.HomePlanet= homePlanet
        self.CryoSleep=cryoSleep
        self.Destination=destination
        self.VIP=vIP
        self.Cabin=cabin
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age":[self.Age],
                "RoomService":[self.RoomService],
                "FoodCourt":[self.FoodCourt],
                "ShoppingMall":[self.ShoppingMall],
                "Spa":[self.Spa],
                "VRDeck":[self.VRDeck],
                "HomePlanet":[self.HomePlanet],
                "CryoSleep":[self.CryoSleep],
                "Destination":[self.Destination],
                "VIP":[self.VIP],
                "Cabin":[self.Cabin]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)