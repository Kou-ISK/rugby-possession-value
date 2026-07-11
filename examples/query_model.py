from rugby_value.model import MarkovEPV
from rugby_value.schema import State

model = MarkovEPV.load("models/premiership-2018-19/model.json")
state = State("Lineout", "5m-Goal (opp)", "Centre", "1")
print(model.value(state))
